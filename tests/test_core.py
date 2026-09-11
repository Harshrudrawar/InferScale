import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inferscale.api import create_app
from inferscale.backends import Event, SGLangBackend, VLLMBackend
from inferscale.decisions import gate, pareto
from inferscale.experiments import manifest, percentile, run_experiment, summarize
from inferscale.models import BenchmarkConfig, ChatRequest, GatePolicy, Message, ModelSpec
from inferscale.runtime import Runtime
from inferscale.storage import Store


@pytest.fixture
def spec():
    return ModelSpec(id="synthetic", engine="synthetic", model_name="fixture", max_inflight=2)


@pytest.fixture
def config():
    return BenchmarkConfig(
        model="synthetic",
        requests=6,
        concurrency=3,
        warmup=0,
        quality_cases=[{"prompt": "ECHO: Paris", "expected": "Paris"}],
    )


@pytest.fixture
def store(tmp_path):
    result = Store(f"sqlite:///{tmp_path}/test.db")
    yield result
    result.close()


async def completed(spec, config, store):
    runtime = Runtime([spec])
    try:
        return await run_experiment(runtime, store, store.create(manifest(config, spec)), config)
    finally:
        await runtime.close()


async def test_experiment_persistence_and_metrics(spec, config, store):
    result = await completed(spec, config, store)
    assert result["status"] == "completed"
    assert result["summary"]["successful_requests"] == 6
    assert result["summary"]["output_tokens_per_second"] > 0
    assert result["quality"]["score"] == 1
    assert result["summary"]["gpu_memory_bytes"] is None
    assert result["samples"][0]["prefill_seconds"] is None
    assert "text" not in result["samples"][0]
    assert result == store.get(result["id"])
    with pytest.raises(ValueError, match="terminal"):
        store.finish(result["id"], "completed", {})


async def test_gate_fails_closed(spec, config, store):
    result = await completed(spec, config, store)
    decision = gate(result)
    assert not decision["passed"]
    assert "Synthetic" in decision["reasons"][0]
    demo = GatePolicy(min_requests=1, allow_synthetic=True)
    assert gate(result, policy=demo)["passed"]
    result["quality"] = None
    assert not gate(result, policy=demo)["passed"]


async def test_mismatched_workload_and_regression(spec, config, store):
    base = await completed(spec, config, store)
    candidate = copy.deepcopy(base)
    policy = GatePolicy(min_requests=1, allow_synthetic=True)
    candidate["summary"]["p95_seconds"] *= 2
    assert "P95 regression exceeds tolerance" in gate(candidate, base, policy)["reasons"]
    candidate["workload_sha256"] = "different"
    assert any("differs" in s for s in gate(candidate, base, policy)["reasons"])


async def test_pareto_dominance_and_mismatch(spec, config, store):
    a = await completed(spec, config, store)
    b = copy.deepcopy(a)
    b["id"] = "better"
    b["summary"]["p95_seconds"] /= 2
    b["summary"]["requests_per_second"] *= 2
    policy = GatePolicy(min_requests=1, allow_synthetic=True)
    assert pareto([a, b], policy) == ["better"]
    b["quality_dataset_sha256"] = "different"
    with pytest.raises(ValueError):
        pareto([a, b], policy)


@pytest.mark.parametrize("backend_cls", [VLLMBackend, SGLangBackend])
async def test_engine_stream_usage_and_chunking(backend_cls):
    async def handler(request):
        body = json.loads(request.content)
        assert body["stream_options"]["include_usage"]
        assert body["model"] == "real-model"
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "multiple tokens in one chunk"}}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 5}},
        ]
        data = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=data)

    backend = backend_cls(
        ModelSpec(id="real", engine="vllm", model_name="real-model", base_url="http://test/v1"),
        transport=httpx.MockTransport(handler),
    )
    try:
        events = [
            e
            async for e in backend.stream(
                ChatRequest(model="real", messages=[Message(role="user", content="hi")])
            )
        ]
        assert events[-1].completion_tokens == 5
        assert events[1].text == "multiple tokens in one chunk"
    finally:
        await backend.close()


async def test_truncated_stream_is_failure():
    backend = VLLMBackend(
        ModelSpec(id="real", engine="vllm", model_name="m", base_url="http://test/v1"),
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, text='data: {"choices": []}\n\n')
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="without"):
            _ = [
                e
                async for e in backend.stream(
                    ChatRequest(model="real", messages=[Message(role="user", content="hi")])
                )
            ]
    finally:
        await backend.close()


async def test_timeout_releases_slot(spec):
    runtime = Runtime([spec])

    class Slow:
        async def stream(self, request):
            await asyncio.sleep(1)
            yield Event(text="late")

        async def close(self):
            pass

    runtime.backends[spec.id] = Slow()
    request = ChatRequest(model=spec.id, messages=[Message(role="user", content="hello")])
    result = await runtime.measure(request, 0.005)
    assert result["error"] == "TimeoutError"
    assert runtime.slots[spec.id]._value == 2
    await runtime.close()


async def test_bounded_concurrency(spec, config, store):
    runtime = Runtime([spec])
    counts = {"active": 0, "peak": 0}

    class Counting:
        async def stream(self, request):
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            try:
                await asyncio.sleep(0.005)
                yield Event(text="ok")
            finally:
                counts["active"] -= 1

        async def close(self):
            pass

    runtime.backends[spec.id] = Counting()
    result = await run_experiment(runtime, store, store.create(manifest(config, spec)), config)
    assert result["status"] == "completed"
    assert counts["peak"] == 2
    assert result["summary"]["output_tokens_per_second"] is None
    await runtime.close()


def test_percentiles_and_failure_denominator():
    assert percentile([1, 2, 3, 4, 5], 0.95) == 4.8
    assert percentile([], 0.95) is None
    sample = {
        "latency_seconds": 1,
        "ttft_seconds": 0.5,
        "tpot_seconds": None,
        "gateway_queue_seconds": 0.1,
        "output_tokens": 1,
        "prompt_tokens": 2,
        "error": None,
    }
    failed = {**sample, "error": "TimeoutError"}
    result = summarize([sample, failed], 2, 3.6)
    assert result["requests_per_second"] == 0.5
    assert result["error_rate"] == 0.5
    assert result["estimated_cost_per_request"] == 0.002
    assert result["tpot_p50_seconds"] is None


def test_recovery_marks_interrupted(spec, config, store):
    record = store.create(manifest(config, spec))
    store.recover()
    assert store.get(record["id"])["status"] == "interrupted"


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        BenchmarkConfig(model="x", concurrency=0)
    with pytest.raises(ValidationError):
        BenchmarkConfig(model="x", prompts=["x" * 65537])
    with pytest.raises(ValidationError):
        ModelSpec(id="x", engine="vllm", model_name="x", base_url="http://user:password@host/v1")


def test_api_chat_auth_stream_and_errors(spec, tmp_path):
    app = create_app([spec], f"sqlite:///{tmp_path}/api.db", api_key="secret")
    headers = {"Authorization": "Bearer secret"}
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/ready", headers=headers).status_code == 200
        body = {
            "model": "synthetic",
            "messages": [{"role": "user", "content": "ECHO: hello world"}],
        }
        response = client.post("/v1/chat/completions", json=body, headers=headers)
        assert response.json()["choices"][0]["message"]["content"] == "hello world"
        assert response.json()["usage"]["completion_tokens"] == 2
        response = client.post(
            "/v1/chat/completions", json={**body, "stream": True}, headers=headers
        )
        assert "data: [DONE]" in response.text
        assert (
            client.post(
                "/v1/chat/completions", json={**body, "model": "bad"}, headers=headers
            ).status_code
            == 404
        )
        assert "inferscale_requests_total" in client.get("/metrics", headers=headers).text


def test_api_experiment_lifecycle(spec, config, tmp_path):
    import time

    app = create_app([spec], f"sqlite:///{tmp_path}/api.db")
    with TestClient(app) as client:
        response = client.post("/experiments", json=config.model_dump())
        assert response.status_code == 202
        id = response.json()["id"]
        for _ in range(100):
            result = client.get(f"/experiments/{id}").json()
            if result["status"] not in {"running", "queued"}:
                break
            time.sleep(0.01)
        assert result["status"] == "completed"
        assert len(client.get("/experiments").json()) == 1
        assert not client.post("/gates", json={"candidate_id": id}).json()["passed"]
        assert client.get("/experiments/missing").status_code == 404
        assert client.post(f"/experiments/{id}/rerun").status_code == 202


def test_completions_and_body_limit(spec, tmp_path):
    app = create_app([spec], f"sqlite:///{tmp_path}/api.db")
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions", json={"model": "synthetic", "prompt": "ECHO: test"}
        )
        assert response.json()["choices"][0]["text"] == "test"
        assert response.json()["object"] == "text_completion"
        response = client.post(
            "/v1/completions", json={"model": "synthetic", "prompt": "ECHO: test", "stream": True}
        )
        assert '"text": "test"' in response.text
        assert client.post("/v1/completions", content=b"x" * (1024 * 1024 + 1)).status_code == 413


async def test_custom_evaluator_rejects_invalid_scores(monkeypatch):
    import sys
    import types

    from inferscale.evaluation import score_cases

    module = types.ModuleType("fake_evaluator")
    module.score = lambda items: {"version": "test", "scores": [float("nan")]}
    monkeypatch.setitem(sys.modules, "fake_evaluator", module)
    items = [{"prompt": "p", "actual": "a", "expected": "a", "error": None}]
    with pytest.raises(ValueError):
        await score_cases(items, "fake_evaluator:score")
    module.score = lambda items: {"version": "test", "scores": [1]}
    items[0]["error"] = "TimeoutError"
    assert (await score_cases(items, "fake_evaluator:score"))["score"] == 0


def test_request_rate_limit(spec, tmp_path):
    from inferscale.limits import RequestLimits

    app = create_app([spec], f"sqlite:///{tmp_path}/api.db")
    app.add_middleware(RequestLimits, per_minute=1)
    with TestClient(app) as client:
        body = {"model": "synthetic", "prompt": "ECHO: x"}
        assert client.post("/v1/completions", json=body).status_code == 200
        response = client.post("/v1/completions", json=body)
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"


async def test_pareto_rejects_different_scorers(spec, config, store):
    a = await completed(spec, config, store)
    b = copy.deepcopy(a)
    b["id"] = "different-scorer"
    b["quality"]["scorer"] = "unrelated:v1"
    with pytest.raises(ValueError, match="scorers"):
        pareto([a, b], GatePolicy(min_requests=1, allow_synthetic=True))


def test_route_explicit_failover(spec, tmp_path):
    from unittest.mock import AsyncMock

    second = spec.model_copy(update={"id": "backup"})
    app = create_app([spec, second], f"sqlite:///{tmp_path}/route.db")
    with TestClient(app) as client:
        runtime = app.state.runtime
        runtime.healthy.update(synthetic=True, backup=True)
        runtime.observed_latency.update(synthetic=0.01, backup=1.0)
        original = runtime.measure

        async def fail_primary(request):
            if request.model == "synthetic":
                return {"error": "ConnectionError"}
            return await original(request)

        runtime.measure = AsyncMock(side_effect=fail_primary)
        body = {
            "request": {
                "model": "synthetic",
                "messages": [{"role": "user", "content": "ECHO: recovered"}],
            },
            "candidates": ["synthetic", "backup"],
        }
        assert client.post("/route", json=body).status_code == 502
        response = client.post("/route", json={**body, "retry_attempts": 1})
        assert response.status_code == 200
        assert response.json()["model"] == "backup"
        assert response.json()["choices"][0]["message"]["content"] == "recovered"


def test_additive_migration_preserves_existing_experiment(spec, config, tmp_path):
    from sqlalchemy import inspect

    from inferscale.operations import migrate

    database = f"sqlite:///{tmp_path}/upgrade.db"
    old_store = Store(database)
    record = old_store.create(manifest(config, spec))
    old_store.close()
    assert migrate(database)["schema_version"] == 2
    assert migrate(database)["schema_version"] == 2
    upgraded = Store(database)
    try:
        assert upgraded.get(record["id"]) == record
        assert {"experiments", "jobs", "studies", "schema_versions"} <= set(
            inspect(upgraded.engine).get_table_names()
        )
    finally:
        upgraded.close()


def test_ray_task_body_without_ray_runtime(spec, config):
    from inferscale.ray_executor import remote_run

    document = manifest(config, spec)
    document["id"] = "remote-body-test"
    result = remote_run(document)
    assert result["status"] == "completed"
    assert result["payload"]["summary"]["successful_requests"] == config.requests
    assert result["payload"]["quality"]["score"] == 1
