"""Integration with Harshrudrawar/OpenEval's actual metric plugin API."""

import hashlib
import inspect
import json
import os
from importlib.metadata import version


def score(items, metric="accuracy"):
    from openeval.infrastructure import metric_plugins

    if metric == "weighted":
        weights = json.loads(
            os.getenv("INFERSCALE_OPENEVAL_WEIGHTS", '{"accuracy":0.5,"contains":0.5}')
        )
    else:
        weights = {metric: 1.0}
    if not weights or any(
        not isinstance(v, (int, float)) or not 0 < v < 1e9 for v in weights.values()
    ):
        raise ValueError("OpenEval weights must be finite and positive")
    judge = json.loads(os.getenv("INFERSCALE_OPENEVAL_JUDGE", "{}"))
    plugins = {name: metric_plugins.build_metric_plugin(name, judge) for name in weights}
    scores = []
    for item in items:
        if item["error"]:
            scores.append(0.0)
            continue
        expected = {"answer": item["expected"]}
        actual = {"output": {"answer": item["actual"]}}
        scores.append(
            sum(
                plugin.evaluate(expected, actual) * weights[name]
                for name, plugin in plugins.items()
            )
            / sum(weights.values())
        )
    fingerprint = hashlib.sha256(
        (
            inspect.getsource(metric_plugins)
            + json.dumps({"weights": weights, "judge": judge}, sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    return {"version": version("openeval") + "-" + fingerprint, "scores": scores}
