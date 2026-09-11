import asyncio
import json
import time
from pathlib import Path

import httpx
from pydantic import Field
from sqlalchemy import Column, Integer, MetaData, Table, select

from .models import StrictModel
from .routing import scaling_decision


class AutoscaleConfig(StrictModel):
    deployment: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    namespace: str = Field(default="inferscale", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    prometheus_url: str
    queue_query: str
    utilization_query: str
    minimum: int = Field(default=1, ge=1)
    maximum: int = Field(default=4, ge=1)
    cooldown_seconds: float = Field(default=120, ge=1)
    interval_seconds: float = Field(default=15, ge=1)


async def kubectl(*args):
    process = await asyncio.create_subprocess_exec(
        "kubectl", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await asyncio.wait_for(process.communicate(), 30)
    if process.returncode:
        raise RuntimeError("kubectl operation failed: " + err.decode()[:200])
    return out.decode()


async def autoscale(config, once=False):
    if config.minimum > config.maximum:
        raise ValueError("minimum must not exceed maximum")
    last_change = 0
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:

        async def metric(query):
            response = await client.get(
                config.prometheus_url.rstrip("/") + "/api/v1/query", params={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            values = data.get("data", {}).get("result", [])
            if data.get("status") != "success" or len(values) != 1:
                raise ValueError("Scaling metric must resolve to exactly one series")
            value = float(values[0]["value"][1])
            if not 0 <= value < 1e12:
                raise ValueError("Scaling metric unavailable or invalid")
            return value

        while True:
            try:
                deployment = json.loads(
                    await kubectl(
                        "-n", config.namespace, "get", "deployment", config.deployment, "-o", "json"
                    )
                )
                current = deployment["spec"]["replicas"]
                depth, util = await asyncio.gather(
                    metric(config.queue_query), metric(config.utilization_query)
                )
                desired = scaling_decision(current, depth, util, config.minimum, config.maximum)
                if desired != current and time.monotonic() - last_change >= config.cooldown_seconds:
                    await kubectl(
                        "-n",
                        config.namespace,
                        "scale",
                        "deployment",
                        config.deployment,
                        "--replicas=" + str(desired),
                        "--current-replicas=" + str(current),
                    )
                    last_change = time.monotonic()
                print(
                    json.dumps(
                        {
                            "current": current,
                            "desired": desired,
                            "queue_depth": depth,
                            "utilization": util,
                        }
                    ),
                    flush=True,
                )
            except (ValueError, RuntimeError, httpx.HTTPError) as exc:
                # Missing metrics never trigger scale-down.
                print(json.dumps({"action": "hold", "error": type(exc).__name__}), flush=True)
            if once:
                return
            await asyncio.sleep(config.interval_seconds)


def migrate(database_url):
    from .jobs import JobQueue
    from .storage import Store
    from .studies import Studies

    store = Store(database_url)
    queue = JobQueue(store)
    Studies(queue, {})
    table = Table("schema_versions", MetaData(), Column("version", Integer, primary_key=True))
    table.metadata.create_all(store.engine)
    with store.engine.begin() as conn:
        versions = set(conn.execute(select(table.c.version)).scalars())
        if 2 not in versions:
            conn.execute(table.insert().values(version=2))
    store.close()
    return {
        "schema_version": 2,
        "strategy": "additive v0.1 to v0.2; existing experiments preserved",
    }


def token_workload(model, revision, lengths, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    prompts = []
    for n in lengths:
        if not 1 <= n <= 65536:
            raise ValueError("Token lengths must be in 1..65536")
        # Validate decoded prompts by re-encoding; no approximate word/token claim.
        text = " The quick brown fox explores inference systems." * (n + 1)
        ids = tokenizer.encode(text, add_special_tokens=False)[:n]
        prompt = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if count != n:
            raise ValueError("Tokenizer roundtrip did not preserve requested length")
        prompts.append({"prompt": prompt, "tokens": count})
    Path(output).write_text(
        json.dumps(
            {
                "model": model,
                "revision": revision,
                "scope": "raw text without special tokens or chat template",
                "prompts": prompts,
            },
            indent=2,
        )
    )
