"""Ray distributes independent benchmark clients; engines are separately deployed."""

import asyncio
import os


def remote_run(document):
    from .experiments import run_experiment
    from .models import BenchmarkConfig, ModelSpec
    from .runtime import Runtime

    class Sink:
        def finish(self, id, status, payload):
            return {"status": status, "payload": payload}

    async def run():
        runtime = Runtime([ModelSpec.model_validate(document["deployment"])])
        try:
            return await run_experiment(
                runtime, Sink(), document, BenchmarkConfig.model_validate(document["config"])
            )
        finally:
            await runtime.close()

    return asyncio.run(run())


async def execute_remote(document, sink):
    import ray

    if not ray.is_initialized():
        ray.init(address=os.getenv("RAY_ADDRESS", "auto"))
    task = ray.remote(num_cpus=1, max_retries=0)(remote_run).remote(document)
    try:
        result = await task
        return sink.finish(document["id"], result["status"], result["payload"])
    except asyncio.CancelledError:
        ray.cancel(task, force=True)
        raise
