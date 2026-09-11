import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from inferscale.api import create_app
from inferscale.deployment import DeploymentProfile
from inferscale.experiments import manifest, run_experiment
from inferscale.jobs import JobQueue, LeaseLost, work_once
from inferscale.models import BenchmarkConfig, GatePolicy, ModelSpec
from inferscale.openeval_bridge import score
from inferscale.optimization import SearchRequest, bayesian_next, candidates, recommend
from inferscale.routing import CircuitBreaker, CircuitOpen, Router, scaling_decision
from inferscale.runtime import Runtime
from inferscale.storage import Store
from inferscale.studies import Studies
from inferscale.telemetry import parse_engine_metrics


@pytest.fixture
def setup(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/platform.db")
    spec = ModelSpec(id="synth", engine="synthetic", model_name="fixture")
    config = BenchmarkConfig(
        model="synth",
        requests=8,
        warmup=0,
        quality_cases=[{"prompt": "ECHO: Paris", "expected": "Paris"}],
    )
    yield store, spec, config
    store.close()


def test_atomic_claim_and_idempotency(setup):
    store, spec, config = setup
    queue = JobQueue(store)
    first = queue.enqueue(config, spec, "one")
    assert queue.enqueue(config, spec, "one")["id"] == first["id"]
    with pytest.raises(ValueError):
        queue.enqueue(config.model_copy(update={"requests": 9}), spec, "one")
    queue.enqueue(config, spec, "two")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: queue.claim(), range(4)))
    assert sum(c is not None for c in claims) == 1
    assert len(store.list()) == 2


def test_fencing_and_expired_lease(setup):
    store, spec, config = setup
    queue = JobQueue(store)
    queue.enqueue(config, spec)
    job = queue.claim()
    with store.engine.begin() as conn:
        conn.execute(update(queue.jobs).where(queue.jobs.c.id == job["id"]).values(expires=0))
    with pytest.raises(LeaseLost):
        queue.finish(job, "completed", {"summary": {}})
    queue.reap()
    assert store.get(job["experiment_id"])["status"] == "interrupted"
    assert queue.claim() is None


async def test_durable_worker_completes(setup):
    store, spec, config = setup
    queue = JobQueue(store)
    job = queue.enqueue(config, spec)
    reopened = JobQueue(store)
    assert await work_once(reopened)
    result = store.get(job["experiment_id"])
    assert result["status"] == "completed"
    assert result["quality"]["score"] == 1
    assert not await work_once(reopened)


async def test_cancellation_releases_resource(setup):
    store, spec, config = setup
    queue = JobQueue(store)
    config = config.model_copy(update={"fault_delay_seconds": 2, "requests": 1})
    job = queue.enqueue(config, spec)
    task = asyncio.create_task(work_once(queue))
    await asyncio.sleep(0.05)
    assert queue.cancel(job["experiment_id"])
    await task
    assert store.get(job["experiment_id"])["status"] == "cancelled"
    assert queue.enqueue(config, spec)
    assert queue.claim() is not None


async def test_open_loop_repetitions_and_overload(setup):
    store, spec, config = setup
    config = config.model_copy(
        update={"concurrency": 1, "requests": 20, "arrival_rate": 10000, "repetitions": 3}
    )
    runtime = Runtime([spec])
    try:
        result = await run_experiment(runtime, store, store.create(manifest(config, spec)), config)
    finally:
        await runtime.close()
    assert result["status"] == "completed"
    assert len(result["samples"]) == 60
    assert len(result["repetitions"]) == 3
    assert any(s["error"] == "LoadGeneratorOverload" for s in result["samples"])
    assert result["summary"]["error_rate"] > 0
    assert result["p95_mean_interval"]["repetitions"] == 3


def test_circuit_recovery_and_scaling():
    breaker = CircuitBreaker(threshold=2, cooldown=0)
    breaker.failure()
    breaker.failure()
    breaker.acquire()
    with pytest.raises(CircuitOpen):
        breaker.acquire()
    breaker.success()
    assert breaker.available()
    assert scaling_decision(2, 40, 0.5, maximum=4) == 4
    assert scaling_decision(2, 0, 0, minimum=1) == 1


async def test_router_skips_open_and_uses_cost(setup):
    _, spec, _ = setup
    other = spec.model_copy(update={"id": "other", "hourly_cost": 1})
    runtime = Runtime([spec, other])
    try:
        router = Router(runtime)
        assert router.choose(["synth", "other"], "low_cost") == "other"
        for _ in range(3):
            runtime.circuits["other"].failure()
        assert router.choose(["synth", "other"]) == "synth"
    finally:
        await runtime.close()


def test_metrics_parse_is_whitelisted():
    values = parse_engine_metrics(
        'vllm:num_requests_waiting{model_name="a"} 2\nvllm:num_requests_waiting{model_name="b"} 3\nsecret_metric 5\n'
    )
    assert values == {"vllm:num_requests_waiting": 5}


def test_actual_openeval_integration():
    pytest.importorskip("openeval")
    items = [
        {"prompt": "capital", "expected": "Paris", "actual": " PARIS ", "error": None},
        {"prompt": "capital", "expected": "Paris", "actual": "London", "error": None},
    ]
    assert score(items, "accuracy")["scores"] == [1, 0]
    items[0]["actual"] = "The capital is Paris."
    assert score(items, "contains")["scores"] == [1, 0]
    items[0]["error"] = "TimeoutError"
    assert score(items, "weighted")["scores"] == [0, 0]


async def test_study_persists_and_recommends(setup):
    store, spec, config = setup
    queue = JobQueue(store)
    studies = Studies(queue, {spec.id: spec})
    request = SearchRequest(
        benchmark=config,
        models=[spec.id],
        concurrencies=[1, 2],
        budget=2,
        policy=GatePolicy(allow_synthetic=True, min_requests=1),
    )
    study = studies.create(request)
    for _ in range(3):
        studies.tick()
        await work_once(queue)
    result = studies.list()[0]
    assert result["id"] == study["id"]
    assert result["status"] == "completed"
    assert result["recommendation"]["recommended_id"] is not None
    records = [store.get(id) for id in result["experiments"].values()]
    assert recommend(records, request.policy, "cost")["recommended_id"] is None


def test_bayesian_search_uses_unseen_trial(setup):
    pytest.importorskip("sklearn")
    _, spec, config = setup
    configs = candidates(
        SearchRequest(benchmark=config, models=[spec.id], concurrencies=[1, 2, 4, 8, 16])
    )
    assert bayesian_next(configs, {0: -1, 1: -0.8, 2: -0.7}) in {3, 4}


def test_deployment_commands_are_arg_lists():
    profile = DeploymentProfile(
        id="qwen",
        engine="vllm",
        model_name="model; harmless",
        max_num_seqs=32,
        tensor_parallel_size=2,
    )
    command = profile.command()
    assert command[command.index("--model") + 1] == "model; harmless"
    assert command[command.index("--max-num-seqs") + 1] == "32"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    other = profile.model_copy(update={"engine": "sglang"})
    assert "--max-running-requests" in other.command()


def test_viewer_role_and_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "INFERSCALE_ACCESS_KEYS",
        json.dumps([{"name": "reader", "role": "viewer", "key": "read-key"}]),
    )
    app = create_app(
        [ModelSpec(id="s", engine="synthetic", model_name="fixture")],
        f"sqlite:///{tmp_path}/api.db",
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer read-key"}
        assert client.get("/registry", headers=headers).status_code == 200
        assert client.post("/experiments", headers=headers, json={"model": "s"}).status_code == 403
        assert client.get("/registry").status_code == 401


def test_study_api_lifecycle(setup, tmp_path):
    _, spec, config = setup
    with TestClient(create_app([spec], f"sqlite:///{tmp_path}/search.db")) as client:
        result = client.post(
            "/studies",
            json={
                "benchmark": config.model_dump(),
                "models": ["synth"],
                "concurrencies": [1, 2],
                "budget": 2,
                "policy": {"allow_synthetic": True, "min_requests": 1},
            },
        )
        assert result.status_code == 202
        for _ in range(100):
            study = client.get("/studies").json()[0]
            if study["status"] != "running":
                break
            time.sleep(0.05)
        assert study["status"] == "completed"
        assert study["recommendation"]["recommended_id"]
        assert len(client.get("/jobs").json()) == 2
        assert client.get("/audit").json()
