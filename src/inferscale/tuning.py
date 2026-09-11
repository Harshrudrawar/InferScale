"""Apply declared engine profiles, benchmark them, and clean up owned processes."""

import json
from pathlib import Path

from .deployment import DeploymentProfile, ProcessManager
from .experiments import manifest, run_experiment
from .models import BenchmarkConfig, ModelSpec
from .runtime import Runtime
from .storage import Store


async def tune(profiles_file, benchmark_file, database, output):
    profiles = [
        DeploymentProfile.model_validate(p) for p in json.loads(Path(profiles_file).read_text())
    ]
    base = BenchmarkConfig.model_validate(json.loads(Path(benchmark_file).read_text()))
    manager = ProcessManager(profiles)
    store = Store(database)
    records = []
    try:
        for profile in profiles:
            await manager.start(profile.id)
            spec = ModelSpec(
                id=profile.id,
                engine=profile.engine,
                model_name=profile.model_name,
                base_url=f"http://127.0.0.1:{profile.port}/v1",
                revision=profile.revision or "unrecorded",
                gpu_count=profile.tensor_parallel_size,
                quantization=profile.quantization or profile.dtype,
                deployment_settings=profile.model_dump(),
                telemetry_local_gpu=True,
            )
            runtime = Runtime([spec])
            try:
                config = base.model_copy(update={"model": profile.id})
                record = store.create(manifest(config, spec))
                records.append(await run_experiment(runtime, store, record, config))
                Path(output).parent.mkdir(exist_ok=True, parents=True)
                Path(output).write_text(json.dumps(records, indent=2))
            finally:
                await runtime.close()
                await manager.stop(profile.id)
    finally:
        await manager.close()
        store.close()
    return records
