"""Isolated authenticated server used by live release checks."""

import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def server():
    with tempfile.TemporaryDirectory() as temp, socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        key = secrets.token_urlsafe(32)
        env = {
            **os.environ,
            "INFERSCALE_DATABASE_URL": f"sqlite:///{temp}/check.db",
            "INFERSCALE_API_KEY": key,
            "INFERSCALE_ACCESS_KEYS": "[]",
            "INFERSCALE_MODELS": str(ROOT / "configs/models.json"),
            "INFERSCALE_EMBEDDED_WORKER": "1",
            "INFERSCALE_EXECUTOR": "local",
        }
        with open(Path(temp) / "server.log", "w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "inferscale.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
            )
            base = f"http://127.0.0.1:{port}"
            try:
                with httpx.Client(trust_env=False, timeout=2) as client:
                    for _ in range(200):
                        if process.poll() is not None:
                            raise RuntimeError("Validation server exited before readiness")
                        try:
                            if client.get(base + "/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(0.1)
                    else:
                        raise RuntimeError("Validation server readiness timed out")
                yield base, key
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
