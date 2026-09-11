"""Externally managed engines; no server-process lifecycle is implied by adapters."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from .models import ChatRequest, CompletionRequest, ModelSpec


@dataclass
class Event:
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class InferenceBackend(Protocol):
    def stream(self, request: ChatRequest) -> AsyncIterator[Event]: ...
    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...


class SyntheticBackend:
    """Deterministic functional fixture, never evidence of LLM/GPU performance."""

    async def stream(self, request):
        content = (
            request.prompt.strip()
            if isinstance(request, CompletionRequest)
            else request.messages[-1].content.strip()
        )
        answer = (
            content[5:].strip()
            if content.startswith("ECHO:")
            else "Synthetic response for local functional testing only."
        )
        words = answer.split()[: request.max_tokens]
        await asyncio.sleep(0.005)
        for i, word in enumerate(words):
            await asyncio.sleep(0.001)
            yield Event(text=(" " if i else "") + word)
        yield Event(
            prompt_tokens=len(request.prompt.split())
            if isinstance(request, CompletionRequest)
            else sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(words),
            finish_reason="length" if len(answer.split()) > request.max_tokens else "stop",
        )

    async def health_check(self):
        return True

    async def close(self):
        pass


class OpenAICompatibleBackend:
    def __init__(self, spec: ModelSpec, transport=None):
        self.spec = spec
        token = os.getenv(spec.api_key_env, "") if spec.api_key_env else ""
        if spec.api_key_env and not token:
            raise ValueError(f"Missing credential environment variable: {spec.api_key_env}")
        self.client = httpx.AsyncClient(
            base_url=spec.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=httpx.Timeout(60, connect=10),
            limits=httpx.Limits(
                max_connections=spec.max_inflight, max_keepalive_connections=spec.max_inflight
            ),
            transport=transport,
            trust_env=False,
        )

    async def stream(self, request):
        body = request.model_dump()
        body.update(model=self.spec.model_name, stream=True, stream_options={"include_usage": True})
        finished = False
        async with self.client.stream(
            "POST",
            "completions" if isinstance(request, CompletionRequest) else "chat/completions",
            json=body,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    finished = True
                    break
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError("Engine returned an error event")
                usage = data.get("usage") or {}
                choices = data.get("choices") or []
                choice = choices[0] if choices else {}
                delta = choice.get("delta") or {}
                # Content chunks are not assumed to correspond 1:1 with tokens.
                yield Event(
                    text=(
                        choice.get("text")
                        if isinstance(request, CompletionRequest)
                        else delta.get("content")
                    )
                    or "",
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=choice.get("finish_reason"),
                )
        if not finished:
            raise RuntimeError("Engine stream ended without [DONE]")

    async def health_check(self):
        try:
            response = await self.client.get("models", timeout=5)
            response.raise_for_status()
            return self.spec.model_name in {m["id"] for m in response.json()["data"]}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False

    async def close(self):
        await self.client.aclose()


class VLLMBackend(OpenAICompatibleBackend):
    pass


class SGLangBackend(OpenAICompatibleBackend):
    pass


def make_backend(spec):
    return {
        "synthetic": lambda _: SyntheticBackend(),
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
        "openai": OpenAICompatibleBackend,
    }[spec.engine](spec)
