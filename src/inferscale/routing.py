import time
from dataclasses import dataclass, field


class CircuitOpen(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown: float = 30
    failures: int = 0
    opened_at: float | None = None
    probe: bool = False

    def available(self):
        return self.opened_at is None or (
            not self.probe and time.monotonic() - self.opened_at >= self.cooldown
        )

    def acquire(self):
        if not self.available():
            raise CircuitOpen("Backend circuit is open")
        if self.opened_at is not None:
            self.probe = True

    def success(self):
        self.failures = 0
        self.opened_at = None
        self.probe = False

    def failure(self):
        self.failures += 1
        self.probe = False
        if self.failures >= self.threshold:
            self.opened_at = time.monotonic()


@dataclass
class Router:
    runtime: object
    latencies: dict = field(default_factory=dict)

    def choose(self, candidates, priority="low_latency"):
        choices = [
            x
            for x in candidates
            if x in self.runtime.specs
            and self.runtime.circuits[x].available()
            and self.runtime.healthy.get(x) is not False
        ]
        if not choices:
            raise CircuitOpen("No eligible backend")

        def score(id):
            spec = self.runtime.specs[id]
            load = self.runtime.inflight.get(id, 0) / spec.max_inflight
            if priority == "low_cost":
                return (spec.hourly_cost is None, spec.hourly_cost or 0, load)
            if priority == "high_quality":
                return (spec.quality_hint is None, -(spec.quality_hint or 0), load)
            if priority == "high_throughput":
                return (load, -spec.max_inflight)
            return (load, self.runtime.observed_latency.get(id, float("inf")))

        return min(choices, key=score)


def scaling_decision(
    current, queue_depth, utilization, minimum=1, maximum=8, target_queue=8, target_utilization=0.8
):
    import math

    desired = max(
        minimum,
        math.ceil(current * utilization / target_utilization),
        math.ceil(queue_depth / target_queue),
    )
    return min(maximum, desired)
