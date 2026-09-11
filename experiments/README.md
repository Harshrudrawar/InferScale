# Optional hardware experiments

No hardware results are bundled. These programs require separately installed GPU tooling.

## CUDA fusion

```bash
nvcc -O3 experiments/cuda/fused_affine_relu.cu -o fused_affine_relu
./fused_affine_relu
```

Compares affine-plus-ReLU in one kernel versus two kernels; verifies numerical agreement and reports event-measured kernel time. Transfers are excluded. For deeper profiling use your installed Nsight tools against the executable. Never generalize this microbenchmark into an LLM serving speedup.

## NCCL collective

With a matching CUDA/PyTorch/NCCL environment and at least two GPUs:

```bash
torchrun --nproc-per-node=2 experiments/nccl_collective.py
```

Runs an all-reduce correctness check and measures collective latency. It does not implement a model-parallel inference engine; use the managed engine tensor-parallel profiles for model-serving studies.

## GPU scaling

After collecting comparable real-engine records with correctly recorded GPU counts:

```bash
python scripts/scaling_report.py results/gpu-runs.json --output results/scaling.json
```

The report calculates throughput speedup divided by GPU-count ratio. Hold workload and quality requirements fixed and compare replication against sharding explicitly.

## Inferentia

See `neuron/README.md`. The generic compatible endpoint adapter can benchmark a prepared server, and `inferscale neuron-compile` implements fixed-shape NeuronX compilation. Compilation and Inf2 execution require the SDK/hardware and are not validated here.
