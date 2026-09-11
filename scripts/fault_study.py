"""Run explicit failure scenarios without altering host networking or external services."""

import asyncio
import json
from pathlib import Path

from inferscale.experiments import manifest, run_experiment
from inferscale.models import BenchmarkConfig, ModelSpec
from inferscale.runtime import Runtime
from inferscale.storage import Store


async def main():
    spec = ModelSpec(id="synthetic", engine="synthetic", model_name="fixture")
    store = Store("sqlite:///fault-study.db")
    scenarios = {
        "baseline": {},
        "transport_failure": {"fault_failure_rate": 0.2},
        "stream_truncation": {"fault_truncate": True},
        "deadline_exceeded": {"fault_delay_seconds": 0.02, "timeout_seconds": 0.005},
        "overload": {"arrival_rate": 10000, "concurrency": 1},
    }
    results = []
    try:
        for name, changes in scenarios.items():
            config = BenchmarkConfig(model=spec.id, requests=20, warmup=0, **changes)
            runtime = Runtime([spec])
            try:
                result = await run_experiment(
                    runtime, store, store.create(manifest(config, spec)), config
                )
            finally:
                await runtime.close()
            results.append(
                {
                    "scenario": name,
                    "status": result["status"],
                    "summary": result.get("summary"),
                    "error_classes": sorted(
                        {s["error"] for s in result.get("samples", []) if s["error"]}
                    ),
                }
            )
        Path("reports/fault-study.json").write_text(
            json.dumps({"synthetic": True, "scenarios": results}, indent=2)
        )
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
