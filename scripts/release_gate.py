import os
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "inferscale.cli",
        "benchmark",
        os.environ["BENCHMARK_CONFIG"],
        "--models",
        os.getenv("INFERSCALE_MODELS", "configs/models.gpu.example.json"),
        "--output",
        "results/candidate.json",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "inferscale.cli",
        "gate",
        "results/candidate.json",
        "--baseline",
        os.environ["BASELINE_PATH"],
        "--policy",
        "configs/policy.release.json",
    ],
    check=True,
)
