"""Print all blocking findings, and ensure the OS inventory was actually scanned."""

import json
import sys
from collections import Counter
from pathlib import Path


def summarize(path):
    report = json.loads(Path(path).read_text())
    results = report.get("Results", [])
    if not any(r.get("Class") == "os-pkgs" for r in results):
        raise ValueError("Security report is missing an OS package scan")
    findings = [v for r in results for v in r.get("Vulnerabilities", [])]
    print(
        json.dumps(
            {
                "os": report.get("Metadata", {}).get("OS"),
                "findings": len(findings),
                "severity": dict(Counter(v["Severity"] for v in findings)),
            }
        )
    )
    for v in findings:
        print(
            f"{v['PkgName']} {v['VulnerabilityID']} {v['Severity']} installed={v['InstalledVersion']} fixed={v.get('FixedVersion') or 'not available'}"
        )
    return findings


if __name__ == "__main__":
    summarize(sys.argv[1])
