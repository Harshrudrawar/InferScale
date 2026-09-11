"""Reproducible, opt-in client-side fault experiments; no arbitrary network destruction."""

import asyncio
import random


class FaultBackend:
    def __init__(self, backend, failure_rate=0, delay_seconds=0, truncate=False, seed=42):
        if not 0 <= failure_rate <= 1 or delay_seconds < 0:
            raise ValueError("Invalid fault settings")
        self.backend = backend
        self.rate = failure_rate
        self.delay = delay_seconds
        self.truncate = truncate
        self.rng = random.Random(seed)

    async def stream(self, request):
        await asyncio.sleep(self.delay)
        if self.rng.random() < self.rate:
            raise ConnectionError("Injected transport failure")
        async for event in self.backend.stream(request):
            yield event
            if self.truncate:
                raise ConnectionError("Injected stream truncation")

    async def health_check(self):
        return await self.backend.health_check()

    async def close(self):
        await self.backend.close()
