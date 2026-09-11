# Real inference setup

Real GPU execution was not available during this build. These are operator setup instructions, not validation claims. Install a compatible serving engine in a separate environment on a supported GPU host. Record your actual engine version, model commit, image digest and hardware. Avoid simultaneous engine launches on a GPU that cannot fit both models.

## vLLM

Example server command (after installing a compatible vLLM build):

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 4096
```

The model must support a chat template for chat requests. Pin `--revision` to a real model commit for reproducibility. Consult the installed engine's `--help` for supported flags and hardware requirements.

## SGLang

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 30000
```

Record the effective dtype and engine settings; do not assume this example enforces every field in the example registry.

## Point InferScale at the server

Copy `configs/models.gpu.example.json` to `configs/models.local.json`, retaining only the running deployment(s). Fill in the actual model revision, engine version, hardware, container digest, quantization and full allocated hourly cost if applicable. The engine must return the configured `model_name` from `/v1/models` for readiness to pass.

For the API on PowerShell:

```powershell
$env:INFERSCALE_MODELS = "configs/models.local.json"
.\.venv\Scripts\inferscale.exe serve
```

For bash:

```bash
export INFERSCALE_MODELS=configs/models.local.json
inferscale serve
```

Use `/ready` to check server/model reachability. Copy the benchmark JSON, change its model alias to `qwen-vllm` or `qwen-sglang`, and replace synthetic ECHO cases with representative model tasks. Then:

```bash
inferscale benchmark configs/benchmark.local.json --models configs/models.local.json --output results/gpu.json
```

Do not run the synthetic ECHO dataset as a meaningful quality benchmark on a real model.

Within a container, `127.0.0.1` refers to that container. Configure the engine's reachable private hostname instead. Never put upstream tokens into registry URLs: use `api_key_env` to name an environment variable. Keep the upstream engine on a trusted network and restrict access.

Registry settings record an existing deployment; editing them does not reconfigure its server. For managed launches, use the deployment profiles and `inferscale tune` described in README.md. That workflow applies engine flags, starts a local process, benchmarks it, and stops it. GPU provisioning and model access remain operator prerequisites.

Sources: [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/), [SGLang sending requests](https://docs.sglang.io/docs/basic_usage/send_request).
