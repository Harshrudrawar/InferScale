# Neuron compilation and serving

`inferscale neuron-compile configs/neuron.example.json --output results/neuron-fixture`

Run from the project root in a supported AWS Neuron SDK environment. The command imports a local model factory, calls `torch_neuronx.trace`, saves `model.pt`, reloads it, and compares its output with the uncompiled module. A manifest records input shapes/dtypes, source and artifact hashes, compiler arguments, model revision, SDK version, elapsed time and verification status. Compilation failures produce a failed manifest and propagate a nonzero exit.

Factories return `(torch.nn.Module, tuple_of_input_tensors)`. The provided example is a small fixed-shape tensor graph, not an LLM. Importing a factory executes local Python code; this is an operator CLI and is never exposed through HTTP. Choose a new output directory for every run.

CPU-only compilation is supported by setting `cpu_backend=true`, `verify=false`, and explicit `compiler_args`, for example `["--target", "trn1"]`. It still requires the Neuron compiler/SDK. An unverified compiled artifact must be validated on its intended accelerator before use.

Tracing fixes shapes and control flow. This compiler integration is not a general autoregressive LLM serving implementation: use a supported Neuron LLM server and register its OpenAI-compatible endpoint with InferScale's `openai` engine for benchmarking, quality evaluation and routing. Record the actual model revision, Neuron runtime, accelerator count and cost.

Source: [AWS PyTorch NeuronX tracing API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html).

Validation here covers compiler orchestration with test doubles and configuration checks. No Neuron SDK compilation or accelerator execution was available.
