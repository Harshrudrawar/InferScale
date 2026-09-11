"""Real localhost HTTP smoke test; starts and cleans up a private server."""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
    env = {
        **os.environ,
        "INFERSCALE_DATABASE_URL": f"sqlite:///{Path(temp) / 'smoke.db'}",
        "INFERSCALE_API_KEY": "smoke-test-key",
        "INFERSCALE_MODELS": str(root / "configs/models.json"),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "inferscale.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "18765",
        ],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(
            base_url="http://127.0.0.1:18765",
            trust_env=False,
            timeout=10,
            headers={"Authorization": "Bearer smoke-test-key"},
        ) as client:
            for _ in range(100):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.ConnectError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("Server did not become ready")
            ui = client.get("/ui/")
            assert ui.status_code == 200 and '<div id="root">' in ui.text
            assets = re.findall(r'(?:src|href)="(/ui/assets/[^"]+)"', ui.text)
            assert assets
            for asset in assets:
                assert client.get(asset).status_code == 200
            reply = client.post(
                "/v1/chat/completions",
                json={
                    "model": "synthetic",
                    "messages": [{"role": "user", "content": "ECHO: End to end works"}],
                },
            ).json()
            assert reply["choices"][0]["message"]["content"] == "End to end works"
            created = client.post(
                "/experiments", json=json.loads((root / "configs/benchmark.json").read_text())
            )
            assert created.status_code == 202
            id = created.json()["id"]
            for _ in range(100):
                result = client.get("/experiments/" + id).json()
                if result["status"] not in {"running", "queued"}:
                    break
                time.sleep(0.02)
            assert result["status"] == "completed"
            assert result["summary"]["successful_requests"] == 32
            assert client.post("/gates", json={"candidate_id": id}).json()["passed"] is False
            assert "inferscale_requests_total" in client.get("/metrics").text
            (root / "reports").mkdir(exist_ok=True)
            (root / "reports/http-smoke.json").write_text(
                json.dumps(
                    {
                        "transport": "real localhost HTTP with uvicorn",
                        "checks": [
                            "health",
                            "dashboard HTML and bundled assets",
                            "authenticated chat",
                            "async experiment",
                            "polling and persistence",
                            "default gate rejects synthetic",
                            "metrics",
                        ],
                        "passed": True,
                        "experiment_id": id,
                    },
                    indent=2,
                )
                + "\n"
            )
            print("Live HTTP smoke passed")
    finally:
        process.terminate()
        process.wait(timeout=10)
