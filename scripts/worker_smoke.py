"""Two actual worker processes share SQL storage. Optionally use a PostgreSQL test schema."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from inferscale.jobs import JobQueue
from inferscale.models import BenchmarkConfig, ModelSpec
from inferscale.operations import migrate
from inferscale.storage import Store


def exercise(database):
    migrate(database)
    store = Store(database)
    workers = []
    try:
        queue = JobQueue(store)
        config = BenchmarkConfig(model="synthetic", requests=12, warmup=0)
        spec = ModelSpec(id="synthetic", engine="synthetic", model_name="fixture")
        jobs = [queue.enqueue(config, spec, f"smoke-{i}") for i in range(4)]
        for _ in range(2):
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-m", "inferscale.cli", "worker"],
                    env={**os.environ, "INFERSCALE_DATABASE_URL": database},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            records = [store.get(j["experiment_id"]) for j in jobs]
            if all(r["status"] == "completed" for r in records):
                assert all(r["summary"]["successful_requests"] == 12 for r in records)
                return {
                    "passed": True,
                    "workers": 2,
                    "jobs": 4,
                    "storage": make_url(database).get_backend_name(),
                    "synthetic": True,
                }
            if any(w.poll() is not None for w in workers):
                raise RuntimeError("A worker exited unexpectedly")
            time.sleep(0.1)
        raise TimeoutError("Workers did not complete queued jobs")
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait(timeout=10)
        store.close()


def main():
    postgres = os.getenv("INFERSCALE_TEST_POSTGRES_URL")
    if postgres:
        # Create and remove only a uniquely owned test schema, never existing tables.
        schema = "inferscale_test_" + uuid4().hex
        admin = create_engine(postgres)
        try:
            with admin.begin() as conn:
                conn.execute(text(f"CREATE SCHEMA {schema}"))
            url = make_url(postgres).update_query_dict({"options": f"-csearch_path={schema}"})
            result = exercise(url.render_as_string(hide_password=False))
        finally:
            with admin.begin() as conn:
                conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            admin.dispose()
    else:
        with tempfile.TemporaryDirectory() as temp:
            result = exercise("sqlite:///" + (Path(temp) / "workers.db").as_posix())
    Path("reports/worker-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
