from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=65536)


class ChatRequest(StrictModel):
    model: str = Field(min_length=1, max_length=128)
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    stream: bool = False

    @model_validator(mode="after")
    def total_size(self):
        if sum(len(m.content) for m in self.messages) > 65536:
            raise ValueError("Total message content exceeds 65536 characters")
        return self


class CompletionRequest(StrictModel):
    model: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=65536)
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    stream: bool = False


class ModelSpec(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    engine: Literal["synthetic", "vllm", "sglang", "openai"]
    model_name: str
    base_url: str | None = None
    api_key_env: str | None = None
    revision: str = "unrecorded"
    engine_version: str = "unrecorded"
    container_digest: str = "unrecorded"
    hardware: str = "unrecorded"
    gpu_count: int = Field(default=0, ge=0)
    quantization: str = "unrecorded"
    deployment_settings: dict = Field(default_factory=dict)
    hourly_cost: float | None = Field(default=None, ge=0)
    resource_group: str = Field(default="default", min_length=1, max_length=128)
    telemetry_local_gpu: bool = False
    metrics_url: str | None = None
    quality_hint: float | None = Field(default=None, ge=0, le=1)
    max_inflight: int = Field(default=16, ge=1, le=1024)

    @model_validator(mode="after")
    def endpoint(self):
        if self.engine != "synthetic":
            parsed = urlparse(self.base_url or "")
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("Real engines require an HTTP(S) base_url ending in /v1")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("Do not put credentials, query strings, or fragments in base_url")
            if not parsed.path.rstrip("/").endswith("/v1"):
                raise ValueError("base_url must end in /v1")
        return self


class Case(StrictModel):
    prompt: str = Field(min_length=1, max_length=65536)
    expected: str = Field(max_length=65536)


class BenchmarkConfig(StrictModel):
    model: str
    requests: int = Field(default=32, ge=1, le=10000)
    concurrency: int = Field(default=4, ge=1, le=256)
    warmup: int = Field(default=2, ge=0, le=100)
    prompts: list[str] = Field(
        default_factory=lambda: ["Explain what an inference server does."],
        min_length=1,
        max_length=1000,
    )
    max_tokens: int = Field(default=64, ge=1, le=8192)
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int = 42
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    repetitions: int = Field(default=1, ge=1, le=30)
    arrival_rate: float | None = Field(default=None, gt=0, le=10000)
    arrival_pattern: Literal["fixed", "poisson"] = "fixed"
    fault_failure_rate: float = Field(default=0, ge=0, le=1)
    fault_delay_seconds: float = Field(default=0, ge=0, le=60)
    fault_truncate: bool = False
    telemetry_interval: float = Field(default=1, ge=0.1, le=60)
    quality_cases: list[Case] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def prompt_size(self):
        if any(not p or len(p) > 65536 for p in self.prompts):
            raise ValueError("Each prompt must have 1..65536 characters")
        return self


class GatePolicy(StrictModel):
    min_quality: float = Field(default=0.9, ge=0, le=1)
    max_error_rate: float = Field(default=0, ge=0, le=1)
    max_p95_seconds: float | None = Field(default=None, gt=0)
    max_latency_regression: float = Field(default=0.1, ge=0)
    max_quality_drop: float = Field(default=0.01, ge=0, le=1)
    min_requests: int = Field(default=100, ge=1)
    allow_synthetic: bool = False
