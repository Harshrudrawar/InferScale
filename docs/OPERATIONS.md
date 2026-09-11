# Operations and safety properties

## Environment

| Variable | Purpose |
| --- | --- |
| `INFERSCALE_DATABASE_URL` | SQLite development database or PostgreSQL URL; API and workers must share it |
| `INFERSCALE_MODELS` | Model registry path for API and direct CLI runs |
| `INFERSCALE_DEPLOYMENTS` | Operator-owned vLLM/SGLang process profiles |
| `INFERSCALE_EMBEDDED_WORKER` | Default `1`; set `0` when running separate workers |
| `INFERSCALE_EXECUTOR` | Embedded worker mode: `local` or `ray` |
| `RAY_ADDRESS` | Address of a prepared Ray cluster |
| `INFERSCALE_API_KEY` | Simple bearer key; unset means local development without auth |
| `INFERSCALE_ACCESS_KEYS` | Role-based array of `{name, role, key}`; overrides the simple key |
| `INFERSCALE_SCORER` | Evaluator reference; configure identically on controller and workers |
| `INFERSCALE_ALLOW_PROCESS_CONTROL` | `1` enables admin process start/stop endpoints |
| `INFERSCALE_OTEL` | `1` enables OpenTelemetry FastAPI instrumentation |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Operator-controlled OTLP collector endpoint |

Roles: `viewer` reads data, `operator` submits/cancels experiments and searches, `admin` additionally controls configured engine processes. Keys stay in the environment. The dashboard holds its key in memory only. Public `/health`, API docs and dashboard assets contain no authenticated experiment data.

POST bodies are capped at 1 MiB, messages and load configs are bounded, and a per-process global POST rate limit is 600/minute. The CLI HTTP server caps concurrent connections at 256. Distributed per-tenant quotas, SSO and externally verified security certification are not supplied.

## Queue and ownership

Jobs and experiment manifests are committed together. An optional `Idempotency-Key` reuses the original submission only when configuration and deployment hashes match. Resource groups serialize benchmarks that share hardware: aliases on the same GPU should use the same group; distinct independent hosts may use separate groups.

Workers claim jobs through SQL compare-and-swap plus a unique resource lease. A lease has a 30-second lifetime and is renewed while work runs. Result writes require the current unexpired owner. Lost workers cannot overwrite a newer terminal state. Queued jobs survive restart; expired running jobs become interrupted and require explicit rerun. They are not automatically retried, because another process may still be using the engine.

Cancellation is cooperative: running jobs become `cancel_requested`, then cancelled after the worker stops its task and releases the lease. A network partition can leave an upstream generation active until its own timeout. SQL fencing prevents stale writes, not physical duplication on a partitioned GPU. Synchronize worker clocks and use an appropriately available PostgreSQL service for multi-host operation.

Search studies persist configuration and trial IDs. Trial submission is idempotent. Study updates use compare-and-swap to prevent stale controller state overwrites. The default is one controller; additional instances still require operational care around managed processes and metrics.

## Inference reliability

The router skips unhealthy/open-circuit candidates. Circuits open after three failures and admit a single probe after cooldown. Callers can explicitly request nonstreaming failover attempts with exponential backoff. No retry occurs after a streaming response has begun. Benchmark attempts never retry, so failures remain visible in metrics.

Process management is for declared local profiles only. Commands use argument arrays, never shell command strings. Only owned processes are stopped on shutdown. Engines are loaded, health-checked, made ready and evicted explicitly. Logs stay in `engine-logs/`; they may contain sensitive engine diagnostics. Other processes using the GPU are outside this manager's ownership.

## Data, audit and migrations

Experiment prompts, expected answers, configs, measured samples and provenance are persisted; raw generated benchmark content is omitted. Audit records cover queue and deployment actions. Terminal records are immutable through the API, not cryptographically append-only; database administrators can edit data.

`inferscale migrate` adds the queue, resource lease, audit, study and schema-version tables and preserves v0.1 experiment records. This release's migration is additive only. Back up the database first; use normal PostgreSQL backups for production. No automatic destructive migration or rollback is executed.

API requests can be traced using OpenTelemetry. Prometheus metrics from the API describe that process's inference traffic and the shared job queue. Distributed worker experiment results are read from SQL; they are not automatically merged into API latency histograms.

Remote hosting requires TLS, protected database access, secret management and backups. The supplied Compose credentials are explicitly local-development values. The optional Kubernetes autoscaler requires working metrics and holds replica count when metrics are missing. Only one autoscaling controller should own a deployment.

## Container security gate

The runtime uses Debian trixie explicitly, applies available package updates, and retains the blocking HIGH/CRITICAL Trivy scan. The image supports Python Ray tasks; optional Ray Java JARs are removed because InferScale never invokes Java tasks and the bundled Java HTTP library had CVE-2026-54399. CI executes an actual Python Ray task in the built image to verify that boundary. Cross-language Java Ray jobs are not supported by this image.

CI retains the complete JSON scan report even when scanning fails. An unfixed distribution vulnerability remains a failing finding; this project does not automatically suppress it. A passing functional test job is not a security qualification.
