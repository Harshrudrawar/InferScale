"""Optional fixed-shape Neuron compilation; install the SDK on a supported host."""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import time
from pathlib import Path

from pydantic import Field

from .models import StrictModel


class NeuronCompileConfig(StrictModel):
    factory: str = Field(pattern=r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")
    model_revision: str = Field(min_length=1)
    compiler_args: list[str] = Field(default_factory=list)
    cpu_backend: bool = False
    verify: bool = True
    rtol: float = Field(default=0.01, ge=0)
    atol: float = Field(default=0.01, ge=0)


def compile_model(config: NeuronCompileConfig, output: str):
    """Factory returns (eval-ready module, tuple of tensors). Never invoked via HTTP."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Choose a new artifact directory; existing results are preserved")
    if config.cpu_backend and config.verify:
        raise ValueError("CPU compilation cannot verify Neuron execution; set verify=false")
    if config.cpu_backend and not any(
        arg == "--target" or arg.startswith("--target=") for arg in config.compiler_args
    ):
        raise ValueError("CPU compilation requires an explicit --target compiler argument")
    try:
        import torch
        import torch_neuronx
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch and the AWS Neuron SDK in a supported environment"
        ) from exc
    module_name, attribute = config.factory.split(":")
    factory = getattr(importlib.import_module(module_name), attribute)
    model, inputs = factory()
    if not isinstance(inputs, tuple) or not inputs or not all(torch.is_tensor(t) for t in inputs):
        raise ValueError("Factory must return (torch module, nonempty tuple of tensors)")
    model.eval()
    destination.mkdir(parents=True)
    source = inspect.getsourcefile(factory)
    report = {
        "config": config.model_dump(),
        "factory_source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source
        else None,
        "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs],
        "torch_version": torch.__version__,
        "neuron_version": importlib.metadata.version("torch-neuronx"),
        "status": "compiling",
        "hardware_verified": False,
    }
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            expected = model(*inputs) if config.verify else None
            traced = torch_neuronx.trace(
                model,
                inputs,
                compiler_args=config.compiler_args,
                compiler_workdir=str(destination / "compiler"),
                cpu_backend=config.cpu_backend,
            )
            artifact = destination / "model.pt"
            torch.jit.save(traced, str(artifact))
            report["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if config.verify:
                loaded = torch.jit.load(str(artifact))
                actual = loaded(*inputs)
                torch.testing.assert_close(actual, expected, rtol=config.rtol, atol=config.atol)
                report["hardware_verified"] = True
            report["status"] = "compiled"
    except Exception as exc:
        report.update(status="failed", error=type(exc).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
