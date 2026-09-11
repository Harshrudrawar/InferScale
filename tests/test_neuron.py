"""Compiler orchestration tests use SDK doubles, not accelerator execution."""

import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inferscale.neuron import NeuronCompileConfig, compile_model


def test_neuron_cpu_requires_target_and_no_hardware_verification(tmp_path):
    config = NeuronCompileConfig(factory="model:create", model_revision="v1", cpu_backend=True)
    with pytest.raises(ValueError, match="cannot verify"):
        compile_model(config, str(tmp_path / "artifact"))
    with pytest.raises(ValueError, match="target"):
        compile_model(config.model_copy(update={"verify": False}), str(tmp_path / "artifact"))


@pytest.mark.parametrize("fails", [False, True])
def test_neuron_manifest_and_verification(monkeypatch, tmp_path, fails):
    from pathlib import Path

    tensor = SimpleNamespace(shape=(4, 16), dtype="float32")
    model = Mock(return_value="expected")

    def factory():
        return model, (tensor,)

    traced = object()
    trace = Mock(return_value=traced)
    assertion = Mock(side_effect=AssertionError("mismatch") if fails else None)
    torch = SimpleNamespace(
        __version__="test-double",
        is_tensor=lambda t: t is tensor,
        inference_mode=contextlib.nullcontext,
        jit=SimpleNamespace(
            save=lambda obj, path: Path(path).write_bytes(b"test-double-artifact"),
            load=lambda path: lambda *args: "actual",
        ),
        testing=SimpleNamespace(assert_close=assertion),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_neuronx", SimpleNamespace(trace=trace))
    monkeypatch.setitem(sys.modules, "test_factory", SimpleNamespace(create=factory))
    monkeypatch.setattr("inferscale.neuron.importlib.metadata.version", lambda name: "test-double")
    config = NeuronCompileConfig(factory="test_factory:create", model_revision="fixture")
    output = tmp_path / "compiled"
    if fails:
        with pytest.raises(AssertionError):
            compile_model(config, str(output))
    else:
        assert compile_model(config, str(output))["status"] == "compiled"
    report = json.loads((output / "manifest.json").read_text())
    assert report["hardware_verified"] is (not fails)
    assert report["status"] == ("failed" if fails else "compiled")
    assert report["inputs"][0]["shape"] == [4, 16]
    assert trace.call_args.kwargs["cpu_backend"] is False
    assertion.assert_called_once_with("actual", "expected", rtol=0.01, atol=0.01)
    with pytest.raises(FileExistsError):
        compile_model(config, str(output))
