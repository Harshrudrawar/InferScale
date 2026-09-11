import random
import statistics


def bootstrap_mean_interval(values, seed=42, draws=1000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {
        "low": means[int(0.025 * draws)],
        "high": means[int(0.975 * draws)],
        "confidence": 0.95,
        "method": "bootstrap mean across repetitions",
        "repetitions": len(values),
    }
