"""Export the actual dashboard with saved evidence; no network requests or fake writes."""

import argparse
import json
from pathlib import Path

from inferscale.api import load_specs
from inferscale.storage import Store


def export(database, output):
    root = Path(__file__).resolve().parents[1]
    store = Store(database)
    try:
        runs = store.list()
    finally:
        store.close()
    data = {
        "/experiments": runs,
        "/registry": [s.model_dump() for s in load_specs(root / "configs/models.json")],
        "/studies": [],
        "/infrastructure": {
            "backends": [],
            "deployments": [],
            "executor": "saved snapshot",
            "process_control_enabled": False,
        },
        "/audit": [],
    }
    for run in runs:
        data["/experiments/" + run["id"]] = run
    payload = json.dumps(data).replace("<", "\\u003c")
    js = (root / "frontend/preview-build/app.js").read_text().replace("</script", "<\\/script")
    css = (root / "frontend/preview-build/app.css").read_text()
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>InferScale — saved project preview</title><style>'
        + css
        + '</style></head><body><div id="root"></div><script>window.__INFERSCALE_SNAPSHOT__='
        + payload
        + ";</script><script>"
        + js
        + "</script></body></html>"
    )
    Path(output).write_text(html)
    print(f"Exported {len(runs)} recorded experiments to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.database, args.output)
