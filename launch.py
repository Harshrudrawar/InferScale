"""Run with Python 3.12+: prepare a local environment, seed real demo runs, open UI."""

import os
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def main():
    if sys.version_info < (3, 12):
        raise SystemExit("Python 3.12 or newer is required. Python 3.12 is the tested version.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.prefix).resolve() != environment.resolve():
        if not python.exists():
            print("Preparing InferScale...", flush=True)
            venv.create(environment, with_pip=True)
        marker = environment / "inferscale-0.3.0-installed"
        if not marker.exists():
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "-c",
                    "requirements.lock",
                    "-e",
                    ".[search,telemetry]",
                ],
                check=True,
            )
            marker.write_text("0.3.0\n")
        raise SystemExit(subprocess.call([str(python), str(Path(__file__).resolve())]))
    import asyncio

    import uvicorn

    from inferscale.api import create_app, load_specs
    from inferscale.demo import seed_demo

    database = "sqlite:///" + (ROOT / "demo.db").as_posix()
    specs = load_specs(str(ROOT / "configs/models.json"))
    asyncio.run(seed_demo(database, specs))
    url = "http://127.0.0.1:8000/ui/"

    def open_when_ready():
        for _ in range(100):
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(url)
                        return
            except OSError:
                time.sleep(0.1)

    print(f"InferScale dashboard: {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(specs, database), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
