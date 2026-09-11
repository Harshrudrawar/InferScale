# InferScale

**A working platform for inference experiments, quality gates, configuration search, and serving operations.**

InferScale v0.3.0 includes the React workspace, Python API, durable SQL queue, distributed execution adapter, managed engine tuning, telemetry collection, OpenEval integration, and deployment packages. Local functional workflows are tested. GPU, Kubernetes and accelerator execution still require suitable infrastructure; this release does not claim hardware performance results.

## Open the final project

For an immediate, read-only look, open `InferScale-preview.html` in a browser. It contains the actual dashboard and recorded synthetic benchmark results. Navigation, charts, filtering, comparisons and evidence inspection work offline. Starting experiments, optimization and deployment actions require the full application.

To run the full application, install Python 3.12, extract this archive, open a terminal in the `inferscale` folder, and run:

```powershell
py -3.12 launch.py
```

On macOS/Linux:

```bash
python3.12 launch.py
```

The launcher creates a local environment, installs the API/search/telemetry dependencies constrained by the lock, runs four real synthetic benchmark configurations into a separate demo database, starts InferScale, and opens your browser. The first launch needs internet access for dependencies. Keep the terminal open; Ctrl+C stops the server. Subsequent launches preserve your experiments. It binds only to your local machine. GPU engines and OpenEval remain optional installations as described below.

## Run the complete local application

Python 3.12 is the tested version. The dashboard is already built in this source archive.

Windows PowerShell, from the extracted `inferscale` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\inferscale.exe migrate
.\.venv\Scripts\inferscale.exe serve
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .
inferscale migrate
inferscale serve
```

Open [the dashboard](http://127.0.0.1:8000/ui/) or [API documentation](http://127.0.0.1:8000/docs). Start with **New experiment**. The default backend is explicitly synthetic: it validates software behavior without downloading a model or allocating a GPU.

The lock includes optional search, telemetry, Ray and OpenEval dependencies for the complete development environment. For a smaller API-only installation use `pip install -e .`; install extras as needed. OpenEval is pinned to your repository commit `a81e15d35cefcaf42b1e01a10e82f43722e6c662`, not an unrelated package of the same name.

## What is implemented

| Area | Implementation |
| --- | --- |
| Dashboard | Overview, measured charts, run submission, comparisons, search studies, model registry, infrastructure, telemetry inspection, JSON export |
| Inference | Streaming/nonstreaming chat and text APIs; synthetic, vLLM, SGLang and generic OpenAI-compatible adapters |
| Durable work | SQL job queue, idempotent submission, atomic resource claims, lease renewal, cancellation and fenced result writes |
| Distributed execution | Independent worker processes and optional Ray remote benchmark execution |
| Workloads | Closed-loop, fixed/Poisson arrival rate, bounded admission, explicit overload errors, repeated trials, bootstrap interval across trial P95 values |
| Quality | Built-in smoke scorer; actual OpenEval accuracy, containment, weighted metrics and judge plugin support |
| Optimization | Resumable grid/random/Bayesian finite-space studies, feasibility gates, quality/latency/throughput Pareto sets, cost and memory constraints |
| Engine tuning | Apply managed vLLM/SGLang profiles with batching, context, dtype/quantization, cache and tensor-parallel settings; benchmark and stop each owned process |
| Telemetry | Prometheus engine metrics; engine-wide phase counter deltas; optional worker-host CPU/RAM/network/GPU/power measurements |
| Routing | Health-aware candidate selection, circuit breakers, cost/quality/latency/load priorities, opt-in nonstreaming failover with backoff |
| Deployment | Docker/Compose, Helm, rendered Kubernetes manifests, optional GPU deployment, HPA, KubeRay resource, Terraform Helm wrapper |
| Operations | Role-based keys, body/rate limits, audit events, additive schema migration, OpenTelemetry instrumentation, metrics-based autoscaler |
| Neuron compilation | Local model factories, NeuronX tracing, artifact manifests, optional output verification on a Neuron device |
| Research utilities | Exact raw-text token workload generation, GPU scaling report, fault study, CUDA fusion experiment and NCCL collective experiment |
| CI | Unit/integration tests, frontend build, live HTTP smoke, container build/security scan, manually triggered hardware regression gate |

## OpenEval

Your real scorer implementation is integrated and tested. Enable it before starting API and workers:

```powershell
$env:INFERSCALE_SCORER = "openeval:weighted"
```

```bash
export INFERSCALE_SCORER=openeval:weighted
```

Supported names: `openeval:accuracy`, `openeval:contains`, `openeval:weighted`, and `openeval:llm_judge`. The judge requires a configured Ollama model. Accuracy/containment do not call a paid model service. See [evaluation](docs/EVALUATION.md).

## CLI workflows

```bash
inferscale benchmark configs/benchmark.json --output results/demo.json
inferscale sweep configs/benchmark.json --concurrency 1 2 4 8 --output results/sweep.json
inferscale gate results/demo.json --policy configs/policy.demo.json
inferscale pareto results/sweep.json --policy configs/policy.demo.json
python scripts/fault_study.py
```

Create search studies in the dashboard or `POST /studies`. They survive controller restarts. Workers process persisted jobs; terminal experiments remain immutable through the API.

To run separate workers, set `INFERSCALE_EMBEDDED_WORKER=0` for the API, then start workers against the same database:

```bash
inferscale worker --database postgresql+psycopg://USER:PASSWORD@HOST/inferscale
```

Prefer `INFERSCALE_DATABASE_URL` to keep credentials out of command history. For a prepared Ray cluster use `RAY_ADDRESS` and `inferscale worker --executor ray`.

## Real engines and hardware

See [GPU setup](docs/GPU_SETUP.md). Example declared profiles are in `configs/deployments.gpu.example.json`. After installing a compatible engine on a GPU host:

```bash
inferscale tune configs/deployments.gpu.example.json configs/benchmark.gpu.json --output results/tuning.json
```

This command starts and stops its own engines. It checks GPU/port conflicts among processes it owns; it cannot discover every other host process or ensure the hardware is free. Pin real model revisions and compatible engine versions before running a study. It does not silently turn synthetic tests into hardware evidence.

## Containers and cluster deployment

```bash
docker compose up --build
```

Compose includes API/dashboard, worker, PostgreSQL, Prometheus and Grafana. Ports bind to loopback. Grafana's local-development login is `admin` / `local_dev_only`; set `GRAFANA_PASSWORD` before first initialization to change it. API: 8000, Grafana: 3000, Prometheus: 9090.

For Kubernetes, create an operator-managed database/key secret, build/push the application image, then install `infrastructure/helm/inferscale`. See [cluster deployment](docs/CLUSTER.md). Helm lint and rendering were verified; no live cluster was available.

## Build and test

```bash
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
npm ci --prefix frontend
npm run build --prefix frontend
python scripts/smoke.py
```

Detailed evidence and environment limitations are in [validation](reports/VALIDATION.md). There is no hosted cloud deployment attached to this archive.

## Important scope boundaries

The software is substantially beyond the original MVP, but **fully validated production operation is not claimed**. Real GPU inference, multi-node Ray, live PostgreSQL/Compose, Kubernetes autoscaling, CUDA/NCCL and Inferentia still need execution in compatible environments. Ray initialization is blocked by process-inspection restrictions in this build runtime. The generic accelerator protocol adapter does not implement Neuron model compilation.

See [scope matrix](docs/ROADMAP.md), [methodology](docs/METHODOLOGY.md), and [operations](docs/OPERATIONS.md). Only publish hardware numbers you actually measure.
