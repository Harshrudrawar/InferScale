"""Local evaluator extension point, including wrappers around a user's OpenEval."""

import asyncio
import importlib
import os

from pydantic import Field

from .models import StrictModel


class Scores(StrictModel):
    version: str = Field(min_length=1)
    scores: list[float]


def scorer_reference():
    return os.getenv("INFERSCALE_SCORER", "builtin:exact_match_strip_v1")


async def score_cases(items, reference):
    if reference == "builtin:exact_match_strip_v1":
        scores = [
            float(not x["error"] and x["actual"].strip() == x["expected"].strip()) for x in items
        ]
        result = Scores(version="1", scores=scores)
    elif reference.startswith("openeval:"):
        from .openeval_bridge import score

        result = Scores.model_validate(
            await asyncio.to_thread(score, items, reference.split(":", 1)[1])
        )
    else:
        module, function = reference.split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        # Trusted local plugin only, configured by operator, never a request field.
        result = Scores.model_validate(await asyncio.to_thread(callback, items))
    if len(result.scores) != len(items) or any(not 0 <= score <= 1 for score in result.scores):
        raise ValueError("Scorer must return exactly one finite [0,1] score per case")
    # Failed generations must never receive credit from a custom evaluator.
    scores = [0.0 if item["error"] else score for item, score in zip(items, result.scores)]
    return {
        "scorer": reference + "@" + result.version,
        "score": sum(scores) / len(scores),
        "cases": len(scores),
        "results": [{"score": score, "error": item["error"]} for item, score in zip(items, scores)],
    }
