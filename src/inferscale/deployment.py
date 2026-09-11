"""Operator-owned inference processes and engine-specific configuration commands."""

import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from .models import StrictModel


class DeploymentProfile(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    engine: Literal["vllm", "sglang"]
    model_name: str = Field(min_length=1)
    revision: str | None = None
    port: int = Field(default=8001, ge=1024, le=65535)
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)
    max_model_len: int = Field(default=4096, ge=128, le=1048576)
    max_num_seqs: int = Field(default=16, ge=1, le=1024)
    gpu_memory_utilization: float = Field(default=0.85, gt=0, lt=1)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    quantization: Literal["awq", "gptq", "fp8", "bitsandbytes"] | None = None
    gpu_devices: list[int] = Field(default_factory=lambda: [0], min_length=1)
    enable_prefix_caching: bool = False

    def command(self):
        if self.engine == "vllm":
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--tensor-parallel-size",
                str(self.tensor_parallel_size),
                "--max-model-len",
                str(self.max_model_len),
                "--max-num-seqs",
                str(self.max_num_seqs),
                "--gpu-memory-utilization",
                str(self.gpu_memory_utilization),
                "--dtype",
                self.dtype,
            ]
            if self.enable_prefix_caching:
                command.append("--enable-prefix-caching")
        else:
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.model_name,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--tp-size",
                str(self.tensor_parallel_size),
                "--context-length",
                str(self.max_model_len),
                "--max-running-requests",
                str(self.max_num_seqs),
                "--mem-fraction-static",
                str(self.gpu_memory_utilization),
                "--dtype",
                self.dtype,
            ]
            if not self.enable_prefix_caching:
                command.append("--disable-radix-cache")
        if self.revision:
            command += ["--revision", self.revision]
        if self.quantization:
            command += ["--quantization", self.quantization]
        return command


class ProcessManager:
    def __init__(self, profiles=(), log_directory="engine-logs"):
        self.profiles = {p.id: p for p in profiles}
        self.processes = {}
        self.states = {p.id: {"state": "cold", "load_seconds": None} for p in profiles}
        self.directory = Path(log_directory)
        self.lock = asyncio.Lock()

    def describe(self):
        result = []
        for id, profile in self.profiles.items():
            process = self.processes.get(id)
            if process and process.returncode is not None:
                self.states[id]["state"] = "failed"
            result.append({"id": id, "profile": profile.model_dump(), **self.states[id]})
        return result

    async def start(self, id, timeout=300):
        async with self.lock:
            profile = self.profiles[id]
            if id in self.processes and self.processes[id].returncode is None:
                return self.states[id]
            for other, p in self.processes.items():
                if p.returncode is None and (
                    self.profiles[other].port == profile.port
                    or set(self.profiles[other].gpu_devices) & set(profile.gpu_devices)
                ):
                    raise ValueError(
                        "Port or GPU already owned by a running profile; stop it first"
                    )
            self.directory.mkdir(exist_ok=True, parents=True)
            self.states[id] = {"state": "loading", "load_seconds": None}
            start = time.monotonic()
            with (self.directory / (id + ".log")).open("ab") as log:
                process = await asyncio.create_subprocess_exec(
                    *profile.command(),
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, profile.gpu_devices)),
                    },
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                )
            self.processes[id] = process
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
                while time.monotonic() - start < timeout:
                    if process.returncode is not None:
                        raise RuntimeError("Engine exited; inspect operator log")
                    try:
                        response = await client.get(f"http://127.0.0.1:{profile.port}/v1/models")
                        if response.is_success and profile.model_name in {
                            m["id"] for m in response.json().get("data", [])
                        }:
                            self.states[id] = {
                                "state": "warm",
                                "load_seconds": time.monotonic() - start,
                            }
                            return self.states[id]
                    except (httpx.HTTPError, ValueError, KeyError):
                        pass
                    await asyncio.sleep(0.5)
            raise TimeoutError("Engine readiness deadline exceeded")
        except BaseException:
            await self.stop(id)
            self.states[id]["state"] = "failed"
            raise

    async def stop(self, id):
        process = self.processes.get(id)
        self.states[id]["state"] = "evicting"
        if process and process.returncode is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 15)
            except asyncio.TimeoutError:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                await process.wait()
        self.states[id]["state"] = "cold"
        return self.states[id]

    async def close(self):
        for id in list(self.processes):
            await self.stop(id)
