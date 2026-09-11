# InferScale 0.3.0 validation

Validated on 2026-09-11 in the build environment with Python 3.12/Linux.

- **38 automated tests passed**, with two warnings (a dependency deprecation and a Gaussian-process kernel bound warning).
- Ruff lint and formatting passed.
- TypeScript compilation and Vite production build passed. The bundled dashboard is included.
- Live localhost HTTP smoke passed using a real uvicorn subprocess: dashboard HTML and assets, authenticated generation, durable experiment submission, polling, SQLite persistence, default gate rejection and Prometheus metrics.
- Actual OpenEval accuracy/contains/weighted plugin integration passed. A separate synthetic end-to-end benchmark used the pinned OpenEval weighted scorer and scored 1.0 on three ECHO fixture cases.
- A 32-request synthetic benchmark and concurrency sweep at 1, 2, 4 and 8 completed. The demo policy passed, the real release policy correctly rejected synthetic evidence with exit code 1, and Pareto selection executed.
- Explicit fault study exercised injected transport failures, truncated streams, deadlines and load-generator overload. Errors remain visible in results rather than disappearing from latency statistics.
- Tests cover job ownership and lease fencing, cancellation, repeated/open-loop runs, study completion and Bayesian selection, role restrictions, failover, incompatible scorer rejection and repeatable additive migration preserving existing experiments.
- Helm lint passed; default and GPU/HPA/Ray-enabled manifests rendered and parsed. Compose YAML parsed. This does not validate a live deployment.
- Ray's benchmark task body passed when called locally. Actual Ray startup failed during managed-runtime process introspection (`psutil.NoSuchProcess`); distributed Ray execution remains unvalidated.

Two real independent worker processes also completed four persisted jobs against temporary SQLite storage (`worker-smoke.json`). PostgreSQL integration now has its own CI job and isolated-schema test script, but could not run here because system package installation is blocked by the managed environment.

The standalone preview includes the actual dashboard bundle and four measured synthetic experiment configurations, each with two repetitions. It is explicitly read-only and makes no API requests. TypeScript and bundled JavaScript checks passed; no browser inspection was available. The launcher bootstrap dependency installation was not run in a fresh Windows environment.

Raw evidence is in this directory. Synthetic timing reflects host scheduling and is **not GPU/LLM performance evidence**. ECHO quality tests validate software integration, not model capability. Missing GPU/engine phase telemetry remains missing, not estimated.

## Not executed here

Real vLLM/SGLang inference with model weights; GPU process tuning; CUDA/NCCL programs; tokenized workload generation with downloaded model assets; multi-node scaling; PostgreSQL service integration; Docker/Compose services; Kubernetes/HPA/KubeRay deployment; Terraform apply; live autoscaling; Prometheus/Grafana services; OTLP export; hosted CI/security scanning; browser visual/accessibility QA; Windows execution; Ollama-based judge scoring.

The NeuronX fixed-shape compilation workflow is implemented. Configuration validation, compiler arguments, artifact manifests and output verification failure handling are tested using SDK doubles; no actual Neuron compilation was performed. See `docs/ROADMAP.md` for the scope matrix.

The smoke script starts and stops its own server. This archive does not deploy a hosted application.
