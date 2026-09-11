"""Exercise the real Ray remote executor on a local CPU Ray instance."""

import asyncio
import json
from pathlib import Path

import ray

from inferscale.experiments import manifest
from inferscale.models import BenchmarkConfig, ModelSpec
from inferscale.ray_executor import execute_remote


class Sink:
    def finish(self, id, status, payload):
        return {"id": id, "status": status, **payload}


def main():
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
        logging_level="ERROR",
    )
    try:
        spec = ModelSpec(id="synthetic", engine="synthetic", model_name="fixture")
        config = BenchmarkConfig(model=spec.id, requests=4, warmup=0)
        document = manifest(config, spec)
        document["id"] = "ray-smoke"
        result = asyncio.run(execute_remote(document, Sink()))
        assert result["status"] == "completed"
        assert result["summary"]["successful_requests"] == 4
        Path("reports").mkdir(exist_ok=True)
        Path("reports/ray-smoke.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "ray_version": ray.__version__,
                    "scope": "Local CPU Ray worker; synthetic backend; not multi-node GPU validation",
                },
                indent=2,
            )
        )
        print("Ray remote execution passed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
