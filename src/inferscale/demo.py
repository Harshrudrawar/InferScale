"""Seed a separate demo database by executing real synthetic benchmark workloads."""

from .experiments import manifest, run_experiment
from .models import BenchmarkConfig
from .runtime import Runtime
from .storage import Store


async def seed_demo(database, specs):
    store = Store(database)
    runtime = Runtime(specs)
    try:
        if store.list():
            return
        synthetic = next((s for s in specs if s.engine == "synthetic"), None)
        if synthetic is None:
            return
        for concurrency in (1, 2, 4, 8):
            config = BenchmarkConfig(
                model=synthetic.id,
                requests=24,
                concurrency=concurrency,
                repetitions=2,
                warmup=2,
                prompts=["ECHO: A reproducible inference experiment"],
                quality_cases=[{"prompt": "ECHO: Paris", "expected": "Paris"}],
            )
            result = await run_experiment(
                runtime, store, store.create(manifest(config, synthetic)), config
            )
            if result["status"] != "completed":
                raise RuntimeError("Demo benchmark failed; inspect the recorded experiment")
    finally:
        await runtime.close()
        store.close()
