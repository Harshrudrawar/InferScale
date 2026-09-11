"""Compare completed same-workload throughput runs at different GPU counts."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("records")
parser.add_argument("--output", default="results/scaling.json")
args = parser.parse_args()
records = json.loads(Path(args.records).read_text())
if any(r["synthetic"] or r["status"] != "completed" for r in records):
    raise ValueError("Scaling requires completed real-engine experiments")
if (
    len(
        {
            (r["workload_sha256"], r["quality_dataset_sha256"], r["measurement_scope"])
            for r in records
        }
    )
    != 1
):
    raise ValueError("Incomparable experiments")
ordered = sorted(records, key=lambda r: r["deployment"]["gpu_count"])
base = ordered[0]
if base["deployment"]["gpu_count"] <= 0 or base["summary"]["requests_per_second"] <= 0:
    raise ValueError("Invalid GPU count or throughput baseline")
report = []
for record in ordered:
    speedup = record["summary"]["requests_per_second"] / base["summary"]["requests_per_second"]
    ideal = record["deployment"]["gpu_count"] / base["deployment"]["gpu_count"]
    report.append(
        {
            "experiment_id": record["id"],
            "gpu_count": record["deployment"]["gpu_count"],
            "speedup": speedup,
            "scaling_efficiency": speedup / ideal,
        }
    )
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2))
