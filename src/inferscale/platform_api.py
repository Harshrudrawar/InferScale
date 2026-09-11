import asyncio
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from .models import ChatRequest, GatePolicy, StrictModel
from .optimization import SearchRequest, recommend
from .routing import Router
from .telemetry import Collector


class RecommendationRequest(StrictModel):
    experiment_ids: list[str] = Field(min_length=1, max_length=256)
    policy: GatePolicy = Field(default_factory=GatePolicy)
    objective: Literal["latency", "throughput", "cost", "balanced"] = "balanced"
    max_memory_bytes: int | None = Field(default=None, gt=0)
    min_throughput: float = Field(default=0, ge=0)


class RouteRequest(StrictModel):
    request: ChatRequest
    candidates: list[str] = Field(min_length=1, max_length=32)
    retry_attempts: int = Field(default=0, ge=0, le=3)
    priority: Literal["low_latency", "high_throughput", "low_cost", "high_quality"] = "low_latency"


def install_platform_api(app, authenticate, generate):
    router = APIRouter(dependencies=[Depends(authenticate)])

    @router.get("/registry")
    async def registry():
        return [s.model_dump() for s in app.state.runtime.specs.values()]

    @router.get("/jobs")
    async def jobs():
        return [
            {k: v for k, v in j.items() if k not in {"payload", "owner", "dedupe"}}
            for j in app.state.queue.list()
        ]

    @router.post("/experiments/{id}/cancel")
    async def cancel(id: str):
        if not app.state.queue.cancel(id):
            raise HTTPException(409, "Job is not cancellable")
        return {"status": "cancellation_requested"}

    @router.get("/studies")
    async def studies():
        return app.state.studies.list()

    @router.post("/studies", status_code=202)
    async def study(request: SearchRequest):
        try:
            return app.state.studies.create(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/recommendations")
    async def recommendations(request: RecommendationRequest):
        records = [app.state.store.get(id) for id in request.experiment_ids]
        if any(r is None for r in records):
            raise HTTPException(404, "Unknown experiment")
        try:
            return recommend(
                records,
                request.policy,
                request.objective,
                request.max_memory_bytes,
                request.min_throughput,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/route")
    async def route(request: RouteRequest):
        from .routing import CircuitOpen

        remaining = list(request.candidates)
        for attempt in range(request.retry_attempts + 1):
            try:
                selected = Router(app.state.runtime).choose(remaining, request.priority)
            except CircuitOpen as exc:
                raise HTTPException(503, str(exc)) from exc
            try:
                return await generate(request.request.model_copy(update={"model": selected}))
            except HTTPException as exc:
                if (
                    exc.status_code != 502
                    or request.request.stream
                    or attempt == request.retry_attempts
                ):
                    raise
                remaining.remove(selected)
                await asyncio.sleep(0.1 * 2**attempt)
        raise HTTPException(503, "No eligible backend")

    @router.get("/infrastructure")
    async def infrastructure():
        runtime = app.state.runtime
        return {
            "backends": [
                {
                    "id": id,
                    "active": runtime.inflight[id],
                    "healthy": runtime.healthy.get(id),
                    "circuit": "closed" if c.opened_at is None else "open",
                    "observed_latency_seconds": runtime.observed_latency.get(id),
                    "failures": c.failures,
                }
                for id, c in runtime.circuits.items()
            ],
            "deployments": app.state.manager.describe(),
            "process_control_enabled": os.getenv("INFERSCALE_ALLOW_PROCESS_CONTROL") == "1",
            "executor": os.getenv("INFERSCALE_EXECUTOR", "local"),
        }

    @router.get("/telemetry/{model}")
    async def telemetry(model: str):
        if model not in app.state.runtime.specs:
            raise HTTPException(404, "Unknown model")
        collector = Collector(app.state.runtime.specs[model])
        await collector.sample()
        return collector.report()

    @router.post("/deployments/{id}/{action}", status_code=202)
    async def deployment(id: str, action: Literal["start", "stop"]):
        if os.getenv("INFERSCALE_ALLOW_PROCESS_CONTROL") != "1":
            raise HTTPException(409, "Process control is disabled; operator must enable it")
        if id not in app.state.manager.profiles:
            raise HTTPException(404, "Unknown deployment profile")
        if id in app.state.deployment_tasks and not app.state.deployment_tasks[id].done():
            raise HTTPException(409, "Deployment action already running")

        async def perform():
            try:
                if action == "start":
                    await app.state.manager.start(id)
                else:
                    await app.state.manager.stop(id)
                app.state.queue.log("deployment." + action, id)
            except Exception as exc:
                app.state.manager.states[id].update(state="failed", error=type(exc).__name__)

        app.state.deployment_tasks[id] = asyncio.create_task(perform())
        return {"id": id, "action": action, "status": "accepted"}

    @router.get("/audit")
    async def audit():
        from sqlalchemy import select

        with app.state.store.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    select(app.state.queue.audit)
                    .order_by(app.state.queue.audit.c.at.desc())
                    .limit(200)
                ).mappings()
            ]

    app.include_router(router)
