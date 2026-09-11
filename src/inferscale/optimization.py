import itertools
import math
import random
from typing import Literal

from pydantic import Field

from .decisions import gate
from .models import BenchmarkConfig, GatePolicy, StrictModel


class SearchRequest(StrictModel):
    benchmark: BenchmarkConfig
    models: list[str] = Field(min_length=1, max_length=32)
    concurrencies: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8], min_length=1, max_length=32
    )
    budget: int = Field(default=8, ge=1, le=256)
    strategy: Literal["grid", "random", "bayesian"] = "grid"
    policy: GatePolicy = Field(default_factory=GatePolicy)
    objective: Literal["latency", "throughput", "cost", "balanced"] = "balanced"
    max_memory_bytes: int | None = Field(default=None, gt=0)
    min_throughput: float = Field(default=0, ge=0)


def candidates(request):
    configs = [
        request.benchmark.model_copy(update={"model": model, "concurrency": concurrency})
        for model, concurrency in itertools.product(request.models, request.concurrencies)
    ]
    configs = [BenchmarkConfig.model_validate(c.model_dump()) for c in configs]
    if request.strategy == "random":
        random.Random(request.benchmark.seed).shuffle(configs)
    return configs


def measured_memory(record):
    # Remote hardware cannot be inferred from benchmark host GPU telemetry.
    if not record.get("deployment", {}).get("telemetry_local_gpu", False):
        return None
    samples = record.get("telemetry", {}).get("samples", [])
    totals = [
        sum(g["memory_used_bytes"] for g in s["local_gpu"] if g["memory_used_bytes"] is not None)
        for s in samples
        if s.get("local_gpu")
    ]
    return max(totals) if totals else None


def recommend(records, policy=None, objective="balanced", max_memory_bytes=None, min_throughput=0):
    policy = policy or GatePolicy()
    scored = []
    excluded = []
    signatures = set()
    for record in records:
        decision = gate(record, policy=policy)
        reasons = list(decision["reasons"])
        summary = record.get("summary", {})
        memory = measured_memory(record)
        if summary.get("requests_per_second", 0) < min_throughput:
            reasons.append("Throughput below target")
        if max_memory_bytes is not None and (memory is None or memory > max_memory_bytes):
            reasons.append("Memory constraint unmet or unmeasured")
        if objective == "cost" and summary.get("estimated_cost_per_request") is None:
            reasons.append("Cost not configured")
        if reasons:
            excluded.append({"id": record["id"], "reasons": reasons})
            continue
        signatures.add(
            (
                record["workload_sha256"],
                record["quality_dataset_sha256"],
                record["synthetic"],
                record["quality"]["scorer"],
                record["measurement_scope"],
            )
        )
        scored.append(record)
    if len(signatures) > 1:
        raise ValueError("Recommendation requires comparable workloads, scorer and evidence type")
    if not scored:
        return {
            "recommended_id": None,
            "ranking": [],
            "excluded": excluded,
            "reason": "No measured configuration satisfies all constraints",
        }
    max_throughput = max(r["summary"]["requests_per_second"] for r in scored) or 1
    max_latency = max(r["summary"]["p95_seconds"] for r in scored) or 1

    def utility(r):
        s = r["summary"]
        if objective == "latency":
            return -s["p95_seconds"]
        if objective == "throughput":
            return s["requests_per_second"]
        if objective == "cost":
            return -s["estimated_cost_per_request"]
        return (
            0.4 * r["quality"]["score"]
            + 0.3 * s["requests_per_second"] / max_throughput
            - 0.3 * s["p95_seconds"] / max_latency
        )

    ranking = sorted(scored, key=utility, reverse=True)
    return {
        "recommended_id": ranking[0]["id"],
        "ranking": [{"id": r["id"], "score": utility(r)} for r in ranking],
        "excluded": excluded,
        "reason": f"Best measured {objective} objective among configurations passing constraints",
        "normalization": "Balanced: 0.4*quality + 0.3*throughput/max - 0.3*p95/max; normalization is within this candidate set",
    }


def bayesian_next(configs, completed, seed=42):
    """Finite-space GP expected improvement; optional scikit-learn dependency."""
    if len(completed) < 3:
        return next(i for i in range(len(configs)) if i not in completed)
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    models = sorted({c.model for c in configs})
    x = np.array(
        [[float(c.model == m) for m in models] + [math.log2(c.concurrency)] for c in configs]
    )
    ids = sorted(completed)
    y = np.array([completed[i] for i in ids])
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(1e-5), normalize_y=True, random_state=seed
    )
    gp.fit(x[ids], y)
    remaining = [i for i in range(len(configs)) if i not in completed]
    mu, sigma = gp.predict(x[remaining], return_std=True)
    best = float(y.max())
    improvements = []
    for mean, std in zip(mu, sigma):
        std = max(float(std), 1e-9)
        z = (float(mean) - best - 0.01) / std
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        improvements.append((mean - best - 0.01) * cdf + std * pdf)
    return remaining[int(np.argmax(improvements))]


def trial_utility(record, request):
    result = recommend(
        [record],
        request.policy,
        request.objective,
        request.max_memory_bytes,
        request.min_throughput,
    )
    if result["recommended_id"] is None:
        return -1e6
    summary = record["summary"]
    if request.objective == "latency":
        return -summary["p95_seconds"]
    if request.objective == "cost":
        return -summary["estimated_cost_per_request"]
    if request.objective == "throughput":
        return summary["requests_per_second"]
    return (
        0.4 * record["quality"]["score"]
        + 0.3 * summary["requests_per_second"] / (1 + summary["requests_per_second"])
        - 0.3 * summary["p95_seconds"] / (1 + summary["p95_seconds"])
    )
