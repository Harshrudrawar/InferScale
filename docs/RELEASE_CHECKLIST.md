# Remaining environment qualification

A green default CI run qualifies the packaged application on a hosted CPU runner. It does not claim real-model throughput, accelerator correctness, or availability of an Internet deployment.

## Repeatable CPU and browser checks

The default workflow runs real localhost HTTP checks, SQL workers against PostgreSQL, the packaged Docker API and Ray task, a blocking vulnerability/secret scan, Chromium dashboard interactions, and 288 HTTP requests at concurrency 1, 8, and 32. Browser screenshots and load measurements are retained in `browser-load-evidence` for review. Local commands:

```bash
python scripts/load_smoke.py
pip install playwright==1.58.0
python -m playwright install --with-deps chromium
python scripts/browser_smoke.py
```

These commands use a private temporary SQLite database and randomly generated session key. No public service or paid model calls are used. The load check verifies responses and readiness; it is not a sustained production capacity test. Browser automation covers core navigation and submission; screenshots still need visual review, and this is not a full accessibility audit.

## Hardware qualification

The manual `hardware qualification` GitHub workflow is ready for prepared self-hosted Linux runners. It is deliberately not dispatched until matching hardware is connected:

- `inferscale-gpu`: two NVIDIA GPUs, matching drivers, CUDA compiler and a CUDA-enabled PyTorch/NCCL installation. Choose `cuda-nccl` to run numerical kernel verification and a two-GPU collective, preserving evidence.
- `inferscale-neuron`: a compatible AWS Neuron accelerator with the Neuron SDK, PyTorch NeuronX, and InferScale dependencies already installed. Choose `neuron` to compile, reload, and verify the fixed-shape example. The workflow does not replace SDK packages with CPU-only packages.

Use dedicated runners for this private repository. The workflow only runs manually, never on pull requests. These microbenchmarks do not substitute for real LLM studies. After the hardware checks, configure an actual pinned model endpoint, run `configs/benchmark.gpu.json`, and evaluate against a real baseline with `scripts/release_gate.py`; the release policy must continue rejecting synthetic evidence.

## Live deployment

Required inputs are the hosting account/cluster, region, budget, image registry, TLS hostname, secret store, database, backup/restore target, and model-serving target. No paid infrastructure has been provisioned and no public URL is claimed.

1. Build and scan a versioned image from the passing commit; push to the chosen private registry and record its digest.
2. Provision PostgreSQL and store `INFERSCALE_DATABASE_URL` and `INFERSCALE_ACCESS_KEYS` in the Kubernetes secret referenced by the chart. Never use the Compose demonstration password.
3. Install the existing Helm chart using the intended image tag and model registry. Apply TLS ingress and restricted network access through the hosting platform.
4. Validate authenticated readiness, unauthorized rejection, experiment completion on separate workers, queue recovery after worker restart, database backup/restore, metrics collection, and rollback to the previous image.
5. Run sustained traffic against the actual models and deployment topology using explicit latency/error objectives; test replica failure and autoscaling before marking production readiness complete.

Hardware runner access and a deployment target are external prerequisites. Passing CPU/browser CI does not mark these steps complete.
