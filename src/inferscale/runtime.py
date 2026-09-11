import asyncio
import time
from contextlib import asynccontextmanager

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .backends import make_backend
from .routing import CircuitBreaker


class Runtime:
    def __init__(self, specs):
        if len({s.id for s in specs}) != len(specs):
            raise ValueError("Duplicate model IDs in registry")
        self.specs = {s.id: s for s in specs}
        self.backends = {s.id: make_backend(s) for s in specs}
        self.slots = {s.id: asyncio.Semaphore(s.max_inflight) for s in specs}
        self.circuits = {s.id: CircuitBreaker() for s in specs}
        self.inflight = {s.id: 0 for s in specs}
        self.observed_latency = {}
        self.healthy = {s.id: None for s in specs}
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "inferscale_requests_total",
            "Completed requests",
            ["model", "status"],
            registry=self.registry,
        )
        self.active = Gauge(
            "inferscale_active_requests",
            "Requests using backend slots",
            ["model"],
            registry=self.registry,
        )
        self.queue = Histogram(
            "inferscale_queue_seconds",
            "Gateway admission wait; not engine queue",
            ["model"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "inferscale_latency_seconds",
            "Gateway end-to-end latency",
            ["model"],
            registry=self.registry,
        )
        self.ttft = Histogram(
            "inferscale_ttft_seconds",
            "Time to first nonempty content chunk",
            ["model"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "inferscale_output_tokens_total",
            "Engine-reported output tokens (synthetic: words)",
            ["model"],
            registry=self.registry,
        )

    @asynccontextmanager
    async def admission(self, model):
        start = time.perf_counter()
        async with self.slots[model]:
            wait = time.perf_counter() - start
            self.queue.labels(model).observe(wait)
            self.active.labels(model).inc()
            self.inflight[model] += 1
            try:
                yield wait
            finally:
                self.active.labels(model).dec()
                self.inflight[model] -= 1

    async def measure(self, request, timeout=60):
        start = time.perf_counter()
        first = last = None
        text = ""
        prompt_tokens = output_tokens = None
        queue = None
        error = None
        finish_reason = None
        try:
            self.circuits[request.model].acquire()
            async with asyncio.timeout(timeout):
                async with self.admission(request.model) as queue:
                    async for event in self.backends[request.model].stream(request):
                        if event.text:
                            now = time.perf_counter()
                            first = first if first is not None else now
                            last = now
                            text += event.text
                        if event.finish_reason is not None:
                            finish_reason = event.finish_reason
                        if event.prompt_tokens is not None:
                            prompt_tokens = event.prompt_tokens
                        if event.completion_tokens is not None:
                            output_tokens = event.completion_tokens
        except asyncio.CancelledError:
            self.circuits[request.model].failure()
            raise
        except Exception as exc:
            # Error bodies may contain prompts or secrets; retain only the class.
            error = type(exc).__name__
        if error:
            if error != "CircuitOpen":
                self.circuits[request.model].failure()
        else:
            self.circuits[request.model].success()
        elapsed = time.perf_counter() - start
        if not error:
            self.observed_latency[request.model] = (
                0.8 * self.observed_latency.get(request.model, elapsed) + 0.2 * elapsed
            )
        self.requests.labels(request.model, "error" if error else "ok").inc()
        self.latency.labels(request.model).observe(elapsed)
        if first is not None:
            self.ttft.labels(request.model).observe(first - start)
        if output_tokens is not None and not error:
            self.tokens.labels(request.model).inc(output_tokens)
        return {
            "latency_seconds": elapsed,
            "gateway_queue_seconds": queue,
            "ttft_seconds": first - start if first is not None else None,
            "tpot_seconds": (last - first) / (output_tokens - 1)
            if first is not None and output_tokens is not None and output_tokens > 1
            else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "text": text,
            "error": error,
            "finish_reason": finish_reason,
            "engine_queue_seconds": None,
            "prefill_seconds": None,
            "decode_seconds": None,
        }

    async def health_loop(self):
        while True:
            for id, backend in self.backends.items():
                try:
                    self.healthy[id] = await backend.health_check()
                except Exception:
                    self.healthy[id] = False
            await asyncio.sleep(10)

    async def close(self):
        await asyncio.gather(*(b.close() for b in self.backends.values()))
