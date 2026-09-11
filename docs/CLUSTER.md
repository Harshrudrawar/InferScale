# Cluster deployment

The chart has been linted and rendered, including optional resources. It has not been installed into a live Kubernetes cluster during this build.

## Prerequisites

- Kubernetes and Helm 3, a PostgreSQL service, and a reachable application image.
- NVIDIA device plugin and suitable GPU nodes for GPU inference.
- KubeRay operator and compatible CRDs if enabling `ray.enabled`.
- Prometheus and a custom-metrics adapter if enabling HPA; install the supplied adapter rule only after engine metrics have namespace/pod labels.

Build/push `Dockerfile` using your registry. The image contains the API, dashboard, worker and locked Ray client dependencies. Use the **same project-containing image** for Ray pods; a stock Ray image does not contain InferScale or your evaluator plugin.

Create a secret named `inferscale-secrets` in the target namespace with `INFERSCALE_DATABASE_URL` and either `INFERSCALE_API_KEY` or `INFERSCALE_ACCESS_KEYS`. Do not commit real secrets. Set evaluator variables in the same secret if desired.

```bash
helm upgrade --install inferscale infrastructure/helm/inferscale \
  --namespace inferscale --create-namespace \
  --set image.repository=YOUR_REGISTRY/inferscale \
  --set image.tag=0.3.0 \
  --set-file registry=configs/models.cluster.json
```

The cluster registry should use reachable service hostnames, for example `http://inferscale-inference:8000/v1`, not a laptop's localhost. The default registry is synthetic so a chart install does not silently download a model.

Enable the optional inference deployment and set its model, pinned revision, compatible image, tensor-parallel GPU count and batch/context settings. Replicas duplicate a deployment; tensor parallelism shards one model across GPUs within a pod. Record both before comparing results. Shared-memory storage is configured for the inference pod.

To enable Ray, set `ray.enabled=true`, `workers.executor=ray`, and `ray.image` to the full project image. The Ray cluster remains private. Native Ray communication is not a public multi-tenant service; protect cluster networking and configure authenticated Ray access according to your installed Ray version.

The chart's HPA uses `vllm_queue_depth` from the custom metrics API, with stabilization windows. `infrastructure/kubernetes/prometheus-adapter-values.yaml` maps the upstream waiting-request metric. HPA is disabled by default. The alternative `inferscale autoscale configs/autoscale.example.json` combines queue depth and GPU utilization through Prometheus and calls kubectl with a replica precondition; do not enable both controllers for the same target.

`infrastructure/terraform` is an optional Helm wrapper for an existing cluster. It does not provision billable cloud GPU resources, networks or managed databases. Terraform apply was not run.

For local multi-service operation, `docker compose up --build` starts PostgreSQL, API, a worker and monitoring. Increase independent workers with `docker compose up --scale worker=2`; resource groups determine whether jobs can execute concurrently. Live Compose validation remains outstanding in this environment.

For a PostgreSQL integration smoke test, set `INFERSCALE_TEST_POSTGRES_URL` to a disposable test database and run `python scripts/worker_smoke.py`. It creates a unique schema, runs two actual workers, verifies four jobs, and removes only that schema. The database user needs schema creation rights. Without that variable it uses temporary SQLite storage.
