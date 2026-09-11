"""Resumable sequential search controller backed by SQL and idempotent jobs."""

import asyncio
import json
from uuid import uuid4

from sqlalchemy import Column, MetaData, String, Table, Text, select, update
from sqlalchemy.exc import IntegrityError

from .optimization import SearchRequest, bayesian_next, candidates, recommend, trial_utility


class Studies:
    def __init__(self, queue, specs):
        self.queue = queue
        self.specs = specs
        self.table = Table(
            "studies",
            MetaData(),
            Column("id", String(36), primary_key=True),
            Column("document", Text, nullable=False),
        )
        self.table.metadata.create_all(queue.engine)

    def create(self, request):
        configs = candidates(request)
        if request.strategy == "bayesian":
            import importlib.util

            if not importlib.util.find_spec("sklearn"):
                raise ValueError("Install inferscale[search] for Bayesian search")
        for c in configs:
            if c.model not in self.specs:
                raise ValueError("Unknown model alias: " + c.model)
        doc = {
            "id": str(uuid4()),
            "status": "running",
            "request": request.model_dump(),
            "experiments": {},
            "recommendation": None,
        }
        with self.queue.engine.begin() as conn:
            conn.execute(self.table.insert().values(id=doc["id"], document=json.dumps(doc)))
        return doc

    def list(self):
        with self.queue.engine.connect() as conn:
            return [json.loads(x) for x in conn.execute(select(self.table.c.document)).scalars()]

    def tick(self):
        for doc in self.list():
            original_document = json.dumps(doc)
            if doc["status"] != "running":
                continue
            request = SearchRequest.model_validate(doc["request"])
            configs = candidates(request)
            records = {int(i): self.queue.store.get(id) for i, id in doc["experiments"].items()}
            if any(
                r["status"] in {"queued", "running", "cancel_requested"} for r in records.values()
            ):
                continue
            if len(records) >= min(request.budget, len(configs)):
                doc["status"] = "completed"
                try:
                    doc["recommendation"] = recommend(
                        list(records.values()),
                        request.policy,
                        request.objective,
                        request.max_memory_bytes,
                        request.min_throughput,
                    )
                except ValueError as exc:
                    doc.update(status="failed", error=str(exc))
            else:
                scores = {i: trial_utility(r, request) for i, r in records.items()}
                index = (
                    bayesian_next(configs, scores, request.benchmark.seed)
                    if request.strategy == "bayesian"
                    else next(i for i in range(len(configs)) if i not in records)
                )
                try:
                    job = self.queue.enqueue(
                        configs[index],
                        self.specs[configs[index].model],
                        key=doc["id"] + ":" + str(index),
                    )
                except IntegrityError:
                    continue
                doc["experiments"][str(index)] = job["experiment_id"]
            with self.queue.engine.begin() as conn:
                conn.execute(
                    update(self.table)
                    .where(self.table.c.id == doc["id"], self.table.c.document == original_document)
                    .values(document=json.dumps(doc))
                )

    async def loop(self):
        while True:
            self.tick()
            await asyncio.sleep(0.5)
