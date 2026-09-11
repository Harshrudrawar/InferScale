# Measurement contract

## Timing and load

The benchmark measures from admission attempt to complete stream consumption using a monotonic clock. It includes gateway admission and adapter-to-engine traffic, but excludes an external client's transport to the InferScale API.

Closed-loop workers submit the next request after the previous finishes. Open-loop fixed or seeded Poisson schedules submit at the requested arrival rate while enforcing an in-flight cap. When that cap is reached, the generator records `LoadGeneratorOverload` instead of silently delaying or dropping the attempt. Scheduler lag is recorded for issued requests. Extremely high rates are bounded by Python/event-loop scheduling and must not be interpreted as a calibrated external load generator.

Warmup runs before the measured repetitions. The pooled summary reports successful-request latency percentiles and throughput across total measured wall time. Errors remain in the attempted-request denominator and consume measured wall time. Repetition summaries are also retained. The bootstrap interval estimates the mean of repetition-level P95 values, not the confidence interval of a pooled P95 or a guarantee of independent trials.

## Metrics

- P50/P95/P99 use sorted linear interpolation on successful-request latencies. Report failure rate beside them to avoid survivor bias.
- TTFT is first nonempty content-chunk latency; buffered chunks, hidden reasoning and network effects can make this differ from true first-token latency.
- TPOT is `(last content timestamp - first content timestamp)/(reported output tokens - 1)`. It is null with fewer than two tokens or missing usage. Multi-token chunks make it an approximation, not a decode kernel timer.
- Token throughput uses engine-reported usage; if any successful request lacks usage, aggregate token throughput is null. Synthetic output counts whitespace words and is labeled accordingly.
- Gateway queue time is not engine scheduler time. Engine queue/prefill/decode interval means come from counter deltas and cover all engine traffic during that interval; they are not attributed to individual benchmark requests.
- GPU/system metrics belong to the worker host. Remote inference GPU memory must not be inferred from client-host measurements. Memory constraints use local GPU samples only when `telemetry_local_gpu=true` explicitly declares co-location. Model-specific attribution still requires a dedicated host/GPU.
- Estimated cost is configured total deployment hourly cost × measured duration / 3600. Load, warmup, evaluation, idle allocation and other billing are excluded. This is a run allocation estimate, not a provider bill.

Missing hardware counters remain null. There are no fabricated GPU metrics. Prometheus counter resets or missing series produce unavailable interval means.

## Inputs and quality

Prompts are explicit text. `inferscale token-workload` uses a pinned tokenizer and verifies encode/decode round trips for raw text without special tokens. Chat templates add tokens; engine-reported prompt usage is the authority for actual chat input lengths.

Quality is evaluated after the load phase. The bundled ECHO tasks and two-question GPU example are smoke fixtures, not representative quality benchmarks. Replace them with a versioned task dataset. A production study needs sufficient task coverage, independent repetitions, randomized run order and controls for temperature, power, thermal state, prefix-cache reuse and co-tenancy.

## Comparisons and search

Regression gates require matching workload/dataset hashes, scorer identity, evidence type and measurement scope. Concurrency and model/deployment selection can vary as experimental dimensions. Fault policy, arrival rate/pattern, generation parameters and repetition count are part of workload identity.

Pareto selection uses quality, P95 and requests/sec after gates. Recommendations can minimize cost or latency, maximize throughput, or use documented normalized balanced weights; optional memory and throughput constraints fail closed when measurements are missing. Search only recommends completed measured candidates, never unmeasured GP predictions.

Bayesian search uses a finite candidate set and Gaussian-process expected improvement. Its balanced surrogate uses bounded quality/latency/throughput terms for cross-trial stability; final balanced ranking normalizes within the feasible measured candidate set. The recommendation is an empirical result for that workload and candidate set, not a universal optimum.

## Reproduction and isolation

Manifests include configuration, model metadata, dataset/workload/source hashes, scorer version, package versions and raw request measurements. Rerun creates a new ID and checks registry/scorer references. It does not restore old containers or guarantee deterministic GPU output.

Resource groups serialize jobs sharing the same declared hardware. Normal API inference may still interfere; use dedicated engines for research runs. SQL fencing protects result ownership, not physical execution under network partition. Recorded hardware/revision metadata is operator-provided and must be checked against the actual deployment before publishing results.
