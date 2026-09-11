"""Collect observed telemetry only; local GPU data is explicitly scoped to its host."""

import asyncio
import os
import shutil
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
}


def parse_engine_metrics(text):
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in METRICS:
                values[sample.name] = values.get(sample.name, 0) + sample.value
    return values


async def gpu_snapshot():
    if not shutil.which("nvidia-smi"):
        return None
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(), 3)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return None
    if process.returncode:
        return None
    result = []
    for line in out.decode().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue

        def number(i, scale=1):
            try:
                return float(parts[i]) * scale
            except ValueError:
                return None

        result.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "utilization_percent": number(2),
                "memory_used_bytes": number(3, 1048576),
                "memory_total_bytes": number(4, 1048576),
                "power_watts": number(5),
                "temperature_c": number(6),
            }
        )
    return result or None


class Collector:
    def __init__(self, spec):
        self.spec = spec
        self.samples = []

    async def sample(self):
        entry = {
            "timestamp": time.time(),
            "local_gpu": await gpu_snapshot(),
            "engine": None,
            "system": None,
        }
        try:
            import psutil

            entry["system"] = {
                "cpu_percent": psutil.cpu_percent(),
                "ram_used_bytes": psutil.virtual_memory().used,
                "network_bytes_sent": psutil.net_io_counters().bytes_sent,
                "network_bytes_recv": psutil.net_io_counters().bytes_recv,
            }
        except ImportError:
            pass
        if self.spec.engine != "synthetic":
            url = (
                self.spec.metrics_url
                or self.spec.base_url.rstrip("/").removesuffix("/v1") + "/metrics"
            )
            key = os.getenv(self.spec.api_key_env, "") if self.spec.api_key_env else ""
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                    response = await client.get(
                        url, headers={"Authorization": "Bearer " + key} if key else {}
                    )
                    response.raise_for_status()
                    entry["engine"] = parse_engine_metrics(response.text)
            except (httpx.HTTPError, ValueError):
                entry["engine_error"] = "MetricsUnavailable"
        self.samples.append(entry)

    async def loop(self, interval):
        while True:
            await self.sample()
            await asyncio.sleep(interval)

    def report(self):
        means = {}
        if len(self.samples) > 1:
            first, last = self.samples[0].get("engine") or {}, self.samples[-1].get("engine") or {}
            for phase in ["queue", "prefill", "decode"]:
                stem = "vllm:request_" + phase + "_time_seconds"
                count = last.get(stem + "_count", 0) - first.get(stem + "_count", 0)
                delta = last.get(stem + "_sum", 0) - first.get(stem + "_sum", 0)
                means[phase + "_seconds"] = delta / count if count > 0 and delta >= 0 else None
        return {
            "samples": self.samples,
            "engine_interval_means": means,
            "scope": "Engine counters cover all traffic; local GPU/system metrics belong to the benchmark worker host",
        }
