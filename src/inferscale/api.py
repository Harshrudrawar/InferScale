import asyncio
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest

from .decisions import gate, pareto
from .deployment import DeploymentProfile, ProcessManager
from .evaluation import scorer_reference
from .jobs import JobQueue, worker_loop
from .limits import RequestLimits
from .models import (
    BenchmarkConfig,
    ChatRequest,
    CompletionRequest,
    GatePolicy,
    ModelSpec,
    StrictModel,
)
from .platform_api import install_platform_api
from .runtime import Runtime
from .storage import Store
from .studies import Studies


class GateRequest(StrictModel):
    candidate_id: str
    baseline_id: str | None = None
    policy: GatePolicy = GatePolicy()


class ParetoRequest(StrictModel):
    experiment_ids: list[str]
    policy: GatePolicy = GatePolicy()


def load_specs(path=None):
    path = path or os.getenv("INFERSCALE_MODELS", "configs/models.json")
    return [ModelSpec.model_validate(s) for s in json.loads(Path(path).read_text())]


def create_app(specs=None, database_url=None, api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = Runtime(specs if specs is not None else load_specs())
        app.state.store = Store(
            database_url or os.getenv("INFERSCALE_DATABASE_URL", "sqlite:///inferscale.db")
        )
        app.state.queue = JobQueue(app.state.store)
        app.state.queue_gauge = Gauge(
            "inferscale_job_queue_depth", "Queued experiments", registry=app.state.runtime.registry
        )
        app.state.studies = Studies(app.state.queue, app.state.runtime.specs)
        profiles_path = Path(os.getenv("INFERSCALE_DEPLOYMENTS", "configs/deployments.json"))
        profiles = (
            [DeploymentProfile.model_validate(p) for p in json.loads(profiles_path.read_text())]
            if profiles_path.exists()
            else []
        )
        app.state.manager = ProcessManager(profiles)
        app.state.deployment_tasks = {}
        app.state.tasks = [
            asyncio.create_task(app.state.studies.loop()),
            asyncio.create_task(app.state.runtime.health_loop()),
        ]
        if os.getenv("INFERSCALE_EMBEDDED_WORKER", "1") == "1":
            app.state.tasks.append(
                asyncio.create_task(
                    worker_loop(app.state.queue, os.getenv("INFERSCALE_EXECUTOR", "local"))
                )
            )
        yield
        for task in app.state.tasks + list(app.state.deployment_tasks.values()):
            task.cancel()
        await asyncio.gather(
            *app.state.tasks, *app.state.deployment_tasks.values(), return_exceptions=True
        )
        await app.state.manager.close()
        await app.state.runtime.close()
        app.state.store.close()

    app = FastAPI(
        title="InferScale",
        version="0.3.0",
        lifespan=lifespan,
        description="Quality-aware inference experiments. Synthetic mode is for functional testing only.",
    )

    app.add_middleware(RequestLimits)

    async def authenticate(request: Request, authorization: str | None = Header(default=None)):
        roles = json.loads(os.getenv("INFERSCALE_ACCESS_KEYS", "[]"))
        if roles:
            principal = next(
                (
                    r
                    for r in roles
                    if secrets.compare_digest(authorization or "", "Bearer " + r["key"])
                ),
                None,
            )
            if principal is None:
                raise HTTPException(401, "Valid bearer token required")
            role = principal["role"]
            if (request.method != "GET" and role == "viewer") or (
                request.url.path.startswith("/deployments/") and role != "admin"
            ):
                raise HTTPException(403, "Insufficient role")
            request.state.actor = principal.get("name", role)
            return
        configured = api_key if api_key is not None else os.getenv("INFERSCALE_API_KEY", "")
        if configured and not secrets.compare_digest(authorization or "", "Bearer " + configured):
            raise HTTPException(401, "Valid bearer token required")

    auth = [Depends(authenticate)]

    def require_model(model):
        if model not in app.state.runtime.specs:
            raise HTTPException(404, "Unknown model alias")

    def record(id):
        result = app.state.store.get(id)
        if result is None:
            raise HTTPException(404, "Unknown experiment")
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.3.0"}

    @app.get("/ready", dependencies=auth)
    async def ready():
        states = {
            key: await backend.health_check() for key, backend in app.state.runtime.backends.items()
        }
        return Response(
            json.dumps({"models": states}),
            status_code=200 if all(states.values()) else 503,
            media_type="application/json",
        )

    @app.get("/metrics", dependencies=auth)
    async def metrics():
        from sqlalchemy import func, select

        with app.state.store.engine.connect() as conn:
            depth = conn.execute(
                select(func.count())
                .select_from(app.state.queue.jobs)
                .where(app.state.queue.jobs.c.status == "queued")
            ).scalar_one()
        app.state.queue_gauge.set(depth)
        return Response(generate_latest(app.state.runtime.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", dependencies=auth)
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": s.id, "object": "model", "owned_by": "inferscale", "engine": s.engine}
                for s in app.state.runtime.specs.values()
            ],
        }

    async def generate(request):
        completion = isinstance(request, CompletionRequest)
        require_model(request.model)
        runtime = app.state.runtime
        id = "chatcmpl-" + str(uuid4())
        created = int(time.time())
        if not request.stream:
            result = await runtime.measure(request)
            if result["error"]:
                raise HTTPException(502, "Inference failed: " + result["error"])
            usage = (
                None
                if result["prompt_tokens"] is None or result["output_tokens"] is None
                else {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["output_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["output_tokens"],
                }
            )
            return {
                "id": id,
                "object": "text_completion" if completion else "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        **(
                            {"text": result["text"]}
                            if completion
                            else {"message": {"role": "assistant", "content": result["text"]}}
                        ),
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": usage,
                "inferscale": {k: v for k, v in result.items() if k != "text"},
            }

        async def events():
            start = time.perf_counter()
            first = None
            status = "ok"
            try:
                runtime.circuits[request.model].acquire()
                async with asyncio.timeout(60):
                    async with runtime.admission(request.model):
                        async for event in runtime.backends[request.model].stream(request):
                            if event.text and first is None:
                                first = time.perf_counter()
                                runtime.ttft.labels(request.model).observe(first - start)
                            data = {
                                "id": id,
                                "object": "text_completion"
                                if completion
                                else "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        **(
                                            {"text": event.text}
                                            if completion
                                            else {
                                                "delta": {"content": event.text}
                                                if event.text
                                                else {}
                                            }
                                        ),
                                        "finish_reason": event.finish_reason,
                                    }
                                ],
                            }
                            if (
                                event.completion_tokens is not None
                                and event.prompt_tokens is not None
                            ):
                                data["usage"] = {
                                    "prompt_tokens": event.prompt_tokens,
                                    "completion_tokens": event.completion_tokens,
                                    "total_tokens": event.prompt_tokens + event.completion_tokens,
                                }
                                runtime.tokens.labels(request.model).inc(event.completion_tokens)
                            yield "data: " + json.dumps(data) + "\n\n"
                runtime.circuits[request.model].success()
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                status = "error"
                if type(exc).__name__ != "CircuitOpen":
                    runtime.circuits[request.model].failure()
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": {
                                "type": type(exc).__name__,
                                "message": "Inference stream failed",
                            }
                        }
                    )
                    + "\n\n"
                )
            finally:
                runtime.requests.labels(request.model, status).inc()
                runtime.latency.labels(request.model).observe(time.perf_counter() - start)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/completions", dependencies=auth)
    async def chat(request: ChatRequest):
        return await generate(request)

    @app.post("/v1/completions", dependencies=auth)
    async def completions(request: CompletionRequest):
        return await generate(request)

    @app.post("/experiments", dependencies=auth, status_code=202)
    async def experiments(
        config: BenchmarkConfig, idempotency_key: str | None = Header(default=None)
    ):
        require_model(config.model)
        if idempotency_key is not None and len(idempotency_key) > 128:
            raise HTTPException(422, "Idempotency key too long")
        try:
            job = app.state.queue.enqueue(
                config, app.state.runtime.specs[config.model], idempotency_key
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"id": job["experiment_id"], "job_id": job["id"], "status": job["status"]}

    @app.get("/experiments", dependencies=auth)
    async def list_experiments(limit: int = Query(default=100, ge=1, le=1000)):
        return [{k: v for k, v in r.items() if k != "samples"} for r in app.state.store.list(limit)]

    @app.get("/experiments/{id}", dependencies=auth)
    async def get_experiment(id: str):
        return record(id)

    @app.post("/experiments/{id}/rerun", dependencies=auth, status_code=202)
    async def rerun(id: str):
        old = record(id)
        config = BenchmarkConfig.model_validate(old["config"])
        require_model(config.model)
        if old.get("scorer_reference") != scorer_reference():
            raise HTTPException(409, "Scorer changed; restore the original scorer before rerun")
        if old["deployment"] != app.state.runtime.specs[config.model].model_dump():
            raise HTTPException(
                409, "Registry deployment has changed; restore recorded deployment before rerun"
            )
        return await experiments(config, None)

    @app.post("/gates", dependencies=auth)
    async def gates(request: GateRequest):
        return gate(
            record(request.candidate_id),
            record(request.baseline_id) if request.baseline_id else None,
            request.policy,
        )

    @app.post("/pareto", dependencies=auth)
    async def frontier(request: ParetoRequest):
        try:
            return {
                "experiment_ids": pareto(
                    [record(id) for id in request.experiment_ids], request.policy
                )
            }
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    install_platform_api(app, authenticate, generate)
    from fastapi.staticfiles import StaticFiles

    frontend = Path(os.getenv("INFERSCALE_FRONTEND", "frontend/dist"))
    if frontend.is_dir():
        app.mount("/ui", StaticFiles(directory=frontend, html=True), name="dashboard")
    from .tracing import instrument

    instrument(app)
    return app


app = create_app()
