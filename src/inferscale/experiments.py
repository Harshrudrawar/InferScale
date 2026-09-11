import asyncio
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from .evaluation import score_cases, scorer_reference
from .faults import FaultBackend
from .models import ChatRequest, Message
from .statistics import bootstrap_mean_interval
from .telemetry import Collector


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def provenance():
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=2
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unrecorded", None
    return {
        "source_sha256": digest(
            {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "inferscale_version": "0.3.0",
        "packages": {p: version(p) for p in ["fastapi", "httpx", "sqlalchemy", "pydantic"]},
        "container_digest": os.getenv("INFERSCALE_IMAGE_DIGEST", "unrecorded"),
    }


def summarize(samples, duration, hourly_cost):
    good = [s for s in samples if not s["error"]]

    def quant(key, q):
        return percentile([s[key] for s in good if s[key] is not None], q)

    output_known = bool(good) and all(s["output_tokens"] is not None for s in good)
    input_known = bool(good) and all(s["prompt_tokens"] is not None for s in good)
    output = sum(s["output_tokens"] for s in good) if output_known else None
    cost = duration * hourly_cost / 3600 if hourly_cost is not None else None
    return {
        "requests": len(samples),
        "successful_requests": len(good),
        "error_rate": 1 - len(good) / len(samples) if samples else 1,
        "duration_seconds": duration,
        "requests_per_second": len(good) / duration,
        "p50_seconds": quant("latency_seconds", 0.5),
        "p95_seconds": quant("latency_seconds", 0.95),
        "p99_seconds": quant("latency_seconds", 0.99),
        "ttft_p50_seconds": quant("ttft_seconds", 0.5),
        "ttft_p95_seconds": quant("ttft_seconds", 0.95),
        "tpot_p50_seconds": quant("tpot_seconds", 0.5),
        "gateway_queue_p95_seconds": quant("gateway_queue_seconds", 0.95),
        "output_tokens_per_second": output / duration if output is not None else None,
        "input_tokens_per_second": sum(s["prompt_tokens"] for s in good) / duration
        if input_known
        else None,
        "token_usage_coverage": sum(s["output_tokens"] is not None for s in good) / len(good)
        if good
        else 0,
        "estimated_run_cost": cost,
        "estimated_cost_per_request": cost / len(good) if cost is not None and good else None,
        "estimated_cost_per_million_output_tokens": cost * 1e6 / output
        if cost is not None and output
        else None,
        "gpu_utilization": None,
        "gpu_memory_bytes": None,
    }


def manifest(config, spec):
    cfg = config.model_dump()
    workload = {
        k: cfg[k]
        for k in [
            "requests",
            "warmup",
            "prompts",
            "max_tokens",
            "temperature",
            "seed",
            "timeout_seconds",
            "repetitions",
            "arrival_rate",
            "arrival_pattern",
            "fault_failure_rate",
            "fault_delay_seconds",
            "fault_truncate",
        ]
    }
    return {
        "scorer_reference": scorer_reference(),
        "config": cfg,
        "deployment": spec.model_dump(),
        "config_sha256": digest(cfg),
        "deployment_sha256": digest(spec.model_dump()),
        "workload_sha256": digest(workload),
        "quality_dataset_sha256": digest(cfg["quality_cases"]),
        "synthetic": spec.engine == "synthetic",
        "provenance": provenance(),
        "measurement_scope": "in-process backend client including gateway admission; excludes HTTP gateway transport",
        "token_units": "whitespace words (synthetic)"
        if spec.engine == "synthetic"
        else "engine-reported tokens",
    }


async def run_experiment(runtime, store, record, config):
    def request(prompt):
        return ChatRequest(
            model=config.model,
            messages=[Message(role="user", content=prompt)],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            seed=config.seed,
        )

    original_backend = runtime.backends[config.model]
    if config.fault_failure_rate or config.fault_delay_seconds or config.fault_truncate:
        runtime.backends[config.model] = FaultBackend(
            runtime.backends[config.model],
            config.fault_failure_rate,
            config.fault_delay_seconds,
            config.fault_truncate,
            config.seed,
        )
    try:
        for i in range(config.warmup):
            warm = await runtime.measure(
                request(config.prompts[i % len(config.prompts)]), config.timeout_seconds
            )
            if warm["error"]:
                raise RuntimeError("Warmup failed: " + warm["error"])
        collector = Collector(runtime.specs[config.model])
        await collector.sample()
        telemetry_task = asyncio.create_task(collector.loop(config.telemetry_interval))
        samples, repetitions = [], []
        duration = 0
        try:
            for repetition in range(config.repetitions):
                run_samples = [None] * config.requests
                indices = iter(range(config.requests))

                async def attempt(i, scheduled=None):
                    sample = await runtime.measure(
                        request(config.prompts[i % len(config.prompts)]), config.timeout_seconds
                    )
                    sample.pop("text")
                    sample.update(request_index=i, repetition=repetition)
                    if scheduled is not None:
                        sample["schedule_lag_seconds"] = max(
                            0, time.perf_counter() - sample["latency_seconds"] - scheduled
                        )
                    run_samples[i] = sample

                async def worker():
                    for i in indices:
                        await attempt(i)

                start = time.perf_counter()
                if config.arrival_rate is None:
                    async with asyncio.TaskGroup() as group:
                        for _ in range(min(config.concurrency, config.requests)):
                            group.create_task(worker())
                else:
                    rng = random.Random(config.seed + repetition)
                    due = start
                    pending = set()
                    async with asyncio.TaskGroup() as group:
                        for i in range(config.requests):
                            await asyncio.sleep(max(0, due - time.perf_counter()))
                            pending = {t for t in pending if not t.done()}
                            if len(pending) >= config.concurrency:
                                run_samples[i] = {
                                    "request_index": i,
                                    "repetition": repetition,
                                    "error": "LoadGeneratorOverload",
                                    "latency_seconds": 0,
                                    "gateway_queue_seconds": None,
                                    "ttft_seconds": None,
                                    "tpot_seconds": None,
                                    "prompt_tokens": None,
                                    "output_tokens": None,
                                    "schedule_lag_seconds": max(0, time.perf_counter() - due),
                                }
                            else:
                                pending.add(group.create_task(attempt(i, due)))
                            due += (
                                rng.expovariate(config.arrival_rate)
                                if config.arrival_pattern == "poisson"
                                else 1 / config.arrival_rate
                            )
                elapsed = time.perf_counter() - start
                duration += elapsed
                samples.extend(run_samples)
                repetitions.append(
                    summarize(run_samples, elapsed, runtime.specs[config.model].hourly_cost)
                )
        finally:
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
        await collector.sample()
        quality = None
        if config.quality_cases:
            items = []
            for case in config.quality_cases:
                result = await runtime.measure(request(case.prompt), config.timeout_seconds)
                items.append(
                    {
                        "prompt": case.prompt,
                        "expected": case.expected,
                        "actual": result["text"],
                        "error": result["error"],
                    }
                )
            quality = await score_cases(items, record["scorer_reference"])
            quality["dataset_sha256"] = record["quality_dataset_sha256"]
        warnings = []
        if config.requests < 100:
            warnings.append("Small sample; p95/p99 are exploratory and not release evidence")
        if record["synthetic"]:
            warnings.append(
                "Synthetic fixture: timings and word counts are not model/GPU benchmarks"
            )
        if any(
            record["deployment"][k] == "unrecorded"
            for k in ["revision", "engine_version", "hardware", "container_digest"]
        ):
            warnings.append("Deployment provenance incomplete; exact reproduction is not assured")
        return store.finish(
            record["id"],
            "completed",
            {
                "summary": summarize(samples, duration, runtime.specs[config.model].hourly_cost),
                "samples": samples,
                "repetitions": repetitions,
                "p95_mean_interval": bootstrap_mean_interval(
                    [r["p95_seconds"] for r in repetitions if r["p95_seconds"] is not None],
                    config.seed,
                ),
                "telemetry": collector.report(),
                "quality": quality,
                "warnings": warnings,
            },
        )
    except asyncio.CancelledError:
        store.finish(record["id"], "interrupted", {"error": "Run cancelled"})
        raise
    except Exception as exc:
        return store.finish(record["id"], "failed", {"error": type(exc).__name__})
    finally:
        runtime.backends[config.model] = original_backend
