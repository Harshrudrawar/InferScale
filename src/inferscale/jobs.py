"""Durable SQL jobs with atomic resource leases and fenced result writes."""

import asyncio
import json
import time
from uuid import uuid4

from sqlalchemy import Column, Float, MetaData, String, Table, Text, select, update
from sqlalchemy.exc import IntegrityError

from .experiments import manifest, run_experiment
from .models import BenchmarkConfig, ModelSpec
from .runtime import Runtime


class LeaseLost(RuntimeError):
    pass


class JobQueue:
    def __init__(self, store):
        self.store = store
        self.engine = store.engine
        meta = MetaData()
        self.jobs = Table(
            "jobs",
            meta,
            Column("id", String(36), primary_key=True),
            Column("experiment_id", String(36), nullable=False),
            Column("status", String(20), nullable=False),
            Column("owner", String(36)),
            Column("expires", Float, default=0),
            Column("resource", String(128), nullable=False),
            Column("dedupe", String(128), unique=True),
            Column("payload", Text, nullable=False),
            Column("created", Float, nullable=False),
        )
        self.locks = Table(
            "resource_leases",
            meta,
            Column("resource", String(128), primary_key=True),
            Column("owner", String(36)),
            Column("expires", Float, nullable=False),
        )
        self.audit = Table(
            "audit_events",
            meta,
            Column("id", String(36), primary_key=True),
            Column("at", Float, nullable=False),
            Column("action", String(128)),
            Column("subject", String(128)),
            Column("actor", String(128)),
        )
        meta.create_all(self.engine)

    def log(self, action, subject, actor="operator"):
        with self.engine.begin() as conn:
            conn.execute(
                self.audit.insert().values(
                    id=str(uuid4()), at=time.time(), action=action, subject=subject, actor=actor
                )
            )

    def enqueue(self, config, spec, key=None):
        document = manifest(config, spec)
        with self.engine.begin() as conn:
            if key:
                previous = (
                    conn.execute(select(self.jobs).where(self.jobs.c.dedupe == key))
                    .mappings()
                    .first()
                )
                if previous:
                    old = json.loads(previous["payload"])
                    if (
                        old["config_sha256"] != document["config_sha256"]
                        or old["deployment_sha256"] != document["deployment_sha256"]
                    ):
                        raise ValueError("Idempotency key already used for a different request")
                    return dict(previous)
            id, experiment_id = str(uuid4()), str(uuid4())
            document.update(
                id=experiment_id,
                status="queued",
                created_at=__import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
            )
            conn.execute(
                self.store.table.insert().values(
                    id=experiment_id,
                    status="queued",
                    created_at=document["created_at"],
                    document=json.dumps(document),
                )
            )
            job = dict(
                id=id,
                experiment_id=experiment_id,
                status="queued",
                owner=None,
                expires=0,
                resource=spec.resource_group,
                dedupe=key,
                payload=json.dumps(document),
                created=time.time(),
            )
            conn.execute(self.jobs.insert().values(**job))
        self.log("experiment.enqueue", experiment_id)
        return job

    def list(self):
        with self.engine.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    select(self.jobs).order_by(self.jobs.c.created.desc()).limit(200)
                ).mappings()
            ]

    def claim(self, lease_seconds=30):
        self.reap()
        now = time.time()
        with self.engine.connect() as conn:
            candidates = list(
                conn.execute(
                    select(self.jobs)
                    .where(self.jobs.c.status == "queued")
                    .order_by(self.jobs.c.created)
                    .limit(100)
                ).mappings()
            )
        for job in candidates:
            owner = str(uuid4())
            try:
                with self.engine.begin() as conn:
                    locked = conn.execute(
                        update(self.locks)
                        .where(
                            self.locks.c.resource == job["resource"], self.locks.c.expires <= now
                        )
                        .values(owner=owner, expires=now + lease_seconds)
                    ).rowcount
                    if not locked:
                        conn.execute(
                            self.locks.insert().values(
                                resource=job["resource"], owner=owner, expires=now + lease_seconds
                            )
                        )
                    claimed = conn.execute(
                        update(self.jobs)
                        .where(self.jobs.c.id == job["id"], self.jobs.c.status == "queued")
                        .values(status="running", owner=owner, expires=now + lease_seconds)
                    ).rowcount
                    if not claimed:
                        raise IntegrityError("Already claimed", {}, None)
                    doc = json.loads(job["payload"])
                    doc["status"] = "running"
                    conn.execute(
                        update(self.store.table)
                        .where(self.store.table.c.id == job["experiment_id"])
                        .values(status="running", document=json.dumps(doc))
                    )
                return {**job, "owner": owner, "status": "running", "expires": now + lease_seconds}
            except IntegrityError:
                continue
        return None

    def renew(self, job, lease_seconds=30):
        now = time.time()
        with self.engine.begin() as conn:
            changed = conn.execute(
                update(self.jobs)
                .where(
                    self.jobs.c.id == job["id"],
                    self.jobs.c.owner == job["owner"],
                    self.jobs.c.status == "running",
                    self.jobs.c.expires > now,
                )
                .values(expires=now + lease_seconds)
            ).rowcount
            if changed != 1:
                return False
            conn.execute(
                update(self.locks)
                .where(self.locks.c.resource == job["resource"], self.locks.c.owner == job["owner"])
                .values(expires=now + lease_seconds)
            )
        return True

    def finish(self, job, status, payload):
        with self.engine.begin() as conn:
            fenced = conn.execute(
                update(self.jobs)
                .where(
                    self.jobs.c.id == job["id"],
                    self.jobs.c.owner == job["owner"],
                    self.jobs.c.status == "running",
                    self.jobs.c.expires > time.time(),
                )
                .values(status=status, expires=0)
            ).rowcount
            if fenced != 1:
                raise LeaseLost("Job ownership expired or job was cancelled")
            doc = json.loads(job["payload"])
            doc.update(
                payload,
                status=status,
                finished_at=__import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
            )
            conn.execute(
                update(self.store.table)
                .where(self.store.table.c.id == job["experiment_id"])
                .values(status=status, document=json.dumps(doc))
            )
            conn.execute(
                update(self.locks)
                .where(self.locks.c.resource == job["resource"], self.locks.c.owner == job["owner"])
                .values(expires=0)
            )
        self.log("experiment." + status, job["experiment_id"])
        return doc

    def cancel(self, experiment_id):
        with self.engine.begin() as conn:
            job = (
                conn.execute(select(self.jobs).where(self.jobs.c.experiment_id == experiment_id))
                .mappings()
                .first()
            )
            if not job or job["status"] not in {"queued", "running"}:
                return False
            status = "cancel_requested" if job["status"] == "running" else "cancelled"
            changed = conn.execute(
                update(self.jobs)
                .where(self.jobs.c.id == job["id"], self.jobs.c.status == job["status"])
                .values(status=status)
            ).rowcount
            if changed != 1:
                return False
            doc = json.loads(job["payload"])
            doc["status"] = status
            conn.execute(
                update(self.store.table)
                .where(self.store.table.c.id == experiment_id)
                .values(status=status, document=json.dumps(doc))
            )
        self.log("experiment.cancel", experiment_id)
        return True

    def acknowledge_cancel(self, job):
        with self.engine.begin() as conn:
            changed = conn.execute(
                update(self.jobs)
                .where(
                    self.jobs.c.id == job["id"],
                    self.jobs.c.owner == job["owner"],
                    self.jobs.c.status == "cancel_requested",
                )
                .values(status="cancelled", expires=0)
            ).rowcount
            if changed:
                doc = json.loads(job["payload"])
                doc["status"] = "cancelled"
                conn.execute(
                    update(self.store.table)
                    .where(self.store.table.c.id == job["experiment_id"])
                    .values(status="cancelled", document=json.dumps(doc))
                )
                conn.execute(
                    update(self.locks).where(self.locks.c.owner == job["owner"]).values(expires=0)
                )

    def reap(self):
        with self.engine.begin() as conn:
            expired = (
                conn.execute(
                    select(self.jobs).where(
                        self.jobs.c.status.in_(["running", "cancel_requested"]),
                        self.jobs.c.expires <= time.time(),
                    )
                )
                .mappings()
                .all()
            )
            for job in expired:
                changed = conn.execute(
                    update(self.jobs)
                    .where(
                        self.jobs.c.id == job["id"],
                        self.jobs.c.expires <= time.time(),
                        self.jobs.c.status.in_(["running", "cancel_requested"]),
                    )
                    .values(status="interrupted")
                ).rowcount
                if changed:
                    doc = json.loads(job["payload"])
                    doc.update(
                        status="interrupted", error="Worker lease expired; explicit rerun required"
                    )
                    conn.execute(
                        update(self.store.table)
                        .where(self.store.table.c.id == job["experiment_id"])
                        .values(status="interrupted", document=json.dumps(doc))
                    )


class FencedStore:
    def __init__(self, queue, job):
        self.queue, self.job = queue, job

    def finish(self, id, status, payload):
        return self.queue.finish(self.job, status, payload)


async def work_once(queue, executor="local"):
    job = queue.claim()
    if not job:
        return False
    document = json.loads(job["payload"])
    runtime = Runtime([ModelSpec.model_validate(document["deployment"])])

    async def execute():
        if executor == "ray":
            from .ray_executor import execute_remote

            return await execute_remote(document, FencedStore(queue, job))
        return await run_experiment(
            runtime,
            FencedStore(queue, job),
            document,
            BenchmarkConfig.model_validate(document["config"]),
        )

    task = asyncio.create_task(execute())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=1)
            if not task.done() and not queue.renew(job):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                queue.acknowledge_cancel(job)
                break
        if task.done() and not task.cancelled():
            try:
                task.result()
            except LeaseLost:
                pass
    except asyncio.CancelledError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    except Exception as exc:
        try:
            queue.finish(job, "failed", {"error": type(exc).__name__})
        except LeaseLost:
            pass
    finally:
        await runtime.close()
    return True


async def worker_loop(queue, executor="local"):
    while True:
        if not await work_once(queue, executor):
            await asyncio.sleep(0.25)
