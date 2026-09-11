"""Fail-closed quality/performance gates and feasible Pareto selection."""

from .models import GatePolicy


def gate(candidate, baseline=None, policy=None):
    policy = policy or GatePolicy()
    reasons = []
    if candidate.get("status") != "completed":
        return {"passed": False, "reasons": ["Candidate is not completed"]}
    summary = candidate["summary"]
    quality = candidate.get("quality")
    if candidate["synthetic"] and not policy.allow_synthetic:
        reasons.append("Synthetic results cannot qualify a real deployment")
    if summary["requests"] < policy.min_requests:
        reasons.append("Insufficient request sample")
    if summary["error_rate"] > policy.max_error_rate:
        reasons.append("Error rate exceeds policy")
    if summary["p95_seconds"] is None:
        reasons.append("No successful latency measurements")
    elif policy.max_p95_seconds is not None and summary["p95_seconds"] > policy.max_p95_seconds:
        reasons.append("P95 latency exceeds SLO")
    if quality is None:
        reasons.append("Quality evaluation is missing")
    elif quality["score"] < policy.min_quality:
        reasons.append("Quality below minimum")
    if baseline:
        comparable = (
            baseline.get("status") == "completed"
            and baseline.get("workload_sha256") == candidate.get("workload_sha256")
            and baseline.get("quality_dataset_sha256") == candidate.get("quality_dataset_sha256")
            and baseline.get("synthetic") == candidate.get("synthetic")
            and baseline.get("measurement_scope") == candidate.get("measurement_scope")
        )
        if not comparable:
            reasons.append("Baseline workload, dataset, measurement scope or evidence type differs")
        else:
            old = baseline["summary"]["p95_seconds"]
            new = summary["p95_seconds"]
            if old is None or old <= 0:
                reasons.append("Baseline has no valid P95")
            elif new is not None and new > old * (1 + policy.max_latency_regression):
                reasons.append("P95 regression exceeds tolerance")
            old_quality = baseline.get("quality")
            if not old_quality or not quality or old_quality["scorer"] != quality["scorer"]:
                reasons.append("Comparable baseline quality is missing")
            elif old_quality["score"] - quality["score"] > policy.max_quality_drop:
                reasons.append("Quality regression exceeds tolerance")
    return {"passed": not reasons, "reasons": reasons, "policy": policy.model_dump()}


def pareto(records, policy=None):
    eligible = [r for r in records if gate(r, policy=policy)["passed"]]
    if (
        eligible
        and len(
            {
                (
                    r["workload_sha256"],
                    r["quality_dataset_sha256"],
                    r["synthetic"],
                    r["measurement_scope"],
                    r["quality"]["scorer"],
                )
                for r in eligible
            }
        )
        > 1
    ):
        raise ValueError(
            "Pareto comparison requires identical workloads, datasets, scorers and measurement scopes"
        )

    def vector(r):
        # Maximize quality and request throughput; minimize latency.
        return (
            r["quality"]["score"],
            r["summary"]["requests_per_second"],
            -r["summary"]["p95_seconds"],
        )

    def dominates(a, b):
        va, vb = vector(a), vector(b)
        return all(x >= y for x, y in zip(va, vb)) and any(x > y for x, y in zip(va, vb))

    return [r["id"] for r in eligible if not any(dominates(other, r) for other in eligible)]
