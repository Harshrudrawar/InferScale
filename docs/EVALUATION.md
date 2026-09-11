# OpenEval and quality evaluation

InferScale now integrates the actual `Harshrudrawar/OpenEval` metric plugin API, pinned to commit `a81e15d35cefcaf42b1e01a10e82f43722e6c662` (OpenEval 1.0.0).

`openeval_bridge.py` calls `build_metric_plugin` and each plugin's `evaluate(expected_output, actual_output)` using the exact structures expected by that repository. It scores the generated responses from InferScale, without regenerating them through another target provider.

Set `INFERSCALE_SCORER` on API and workers:

| Value | Behavior |
| --- | --- |
| `builtin:exact_match_strip_v1` | Local fixture scorer, trim-only exact match |
| `openeval:accuracy` | OpenEval case/whitespace-normalized structural accuracy |
| `openeval:contains` | OpenEval short-answer containment |
| `openeval:weighted` | Weighted combination; default 50% accuracy, 50% contains |
| `openeval:llm_judge` | OpenEval Ollama judge; requires a reachable configured model |

Set `INFERSCALE_OPENEVAL_WEIGHTS` to JSON such as `{"accuracy":0.25,"contains":0.75}`. Set `INFERSCALE_OPENEVAL_JUDGE` to JSON such as `{"provider":"ollama","model":"llama3","base_url":"http://localhost:11434/api"}` when using a judge. No judge was run during build validation.

Scorer identity includes the installed OpenEval version and a hash of metric source and settings. Each dataset is separately hashed. Failed target generations receive zero credit in InferScale; this is an intentional stricter denominator than excluding failed cases. Invalid or missing evaluator output fails the experiment. Exact match and ECHO fixtures are not comprehensive LLM quality evaluations.

Trusted custom `module:function` callbacks remain supported. They receive `{prompt, expected, actual, error}` items and return `{"version":"...","scores":[...]}` with one finite [0,1] score per item. Configure plugins only on the operator side. Networked plugin calls need their own timeout; Python thread execution is not forcibly killable.

Integration tests exercised the actual OpenEval accuracy, containment and weighted scorers. Source: [OpenEval metric plugins](https://github.com/Harshrudrawar/OpenEval/blob/a81e15d35cefcaf42b1e01a10e82f43722e6c662/openeval/infrastructure/metric_plugins.py).
