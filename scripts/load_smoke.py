"""Real HTTP concurrency qualification; synthetic backend, not GPU evidence."""

import asyncio
import json
import time

import httpx
from validation_server import ROOT, server


async def check(base, key):
    phases = []
    async with httpx.AsyncClient(
        base_url=base, trust_env=False, timeout=30, headers={"Authorization": f"Bearer {key}"}
    ) as client:
        assert (
            await client.get("/registry", headers={"Authorization": "Bearer invalid"})
        ).status_code == 401
        for concurrency in (1, 8, 32):
            semaphore = asyncio.Semaphore(concurrency)
            durations = []

            async def request(index):
                async with semaphore:
                    started = time.perf_counter()
                    response = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "synthetic",
                            "messages": [{"role": "user", "content": f"ECHO: request-{index}"}],
                        },
                    )
                    response.raise_for_status()
                    assert response.json()["choices"][0]["message"]["content"] == f"request-{index}"
                    durations.append(time.perf_counter() - started)

            started = time.perf_counter()
            await asyncio.gather(*(request(i) for i in range(96)))
            elapsed = time.perf_counter() - started
            durations.sort()
            phases.append(
                {
                    "concurrency": concurrency,
                    "requests": len(durations),
                    "errors": 0,
                    "elapsed_seconds": elapsed,
                    "requests_per_second": len(durations) / elapsed,
                    "p95_seconds": durations[int(0.95 * (len(durations) - 1))],
                }
            )
        assert (await client.get("/ready")).status_code == 200
        assert (await client.get("/metrics")).status_code == 200
    return {
        "passed": True,
        "scope": "localhost HTTP, synthetic backend; not production capacity or GPU evidence",
        "phases": phases,
    }


if __name__ == "__main__":
    with server() as (base, key):
        report = asyncio.run(check(base, key))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/load-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
