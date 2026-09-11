import argparse
import asyncio
import json
import os
from pathlib import Path

from .api import load_specs
from .decisions import gate, pareto
from .experiments import manifest, run_experiment
from .models import BenchmarkConfig, GatePolicy
from .runtime import Runtime
from .storage import Store


async def execute(args):
    runtime = Runtime(load_specs(args.models))
    store = Store(args.database)
    records = []
    try:
        raw = json.loads(Path(args.config).read_text())
        if args.command == "sweep":
            configs = [
                BenchmarkConfig.model_validate({**raw, "concurrency": c}) for c in args.concurrency
            ]
        else:
            configs = [BenchmarkConfig.model_validate(raw)]
        for config in configs:
            if config.model not in runtime.specs:
                raise ValueError("Unknown model alias")
            record = store.create(manifest(config, runtime.specs[config.model]))
            completed = await run_experiment(runtime, store, record, config)
            records.append(completed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(records if args.command == "sweep" else records[0], indent=2) + "\n"
        )
        for r in records:
            print(
                json.dumps(
                    {
                        "id": r["id"],
                        "status": r["status"],
                        "synthetic": r["synthetic"],
                        "summary": r.get("summary"),
                    }
                )
            )
        return (
            0
            if all(r["status"] == "completed" and r["summary"]["error_rate"] == 0 for r in records)
            else 1
        )
    finally:
        await runtime.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(prog="inferscale")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["benchmark", "sweep"]:
        p = subs.add_parser(name)
        p.add_argument("config")
        p.add_argument("--models", default="configs/models.json")
        p.add_argument(
            "--database", default=os.getenv("INFERSCALE_DATABASE_URL", "sqlite:///inferscale.db")
        )
        p.add_argument("--output", default="results/latest.json")
        if name == "sweep":
            p.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    p = subs.add_parser("gate")
    p.add_argument("candidate")
    p.add_argument("--baseline")
    p.add_argument("--policy")
    p = subs.add_parser("pareto")
    p.add_argument("results")
    p.add_argument("--policy")
    p = subs.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = subs.add_parser("worker")
    p.add_argument(
        "--database", default=os.getenv("INFERSCALE_DATABASE_URL", "sqlite:///inferscale.db")
    )
    p.add_argument("--executor", choices=["local", "ray"], default="local")
    p.add_argument("--once", action="store_true")
    p = subs.add_parser("migrate")
    p.add_argument(
        "--database", default=os.getenv("INFERSCALE_DATABASE_URL", "sqlite:///inferscale.db")
    )
    p = subs.add_parser("autoscale")
    p.add_argument("config")
    p.add_argument("--once", action="store_true")
    p = subs.add_parser("tune")
    p.add_argument("profiles")
    p.add_argument("benchmark")
    p.add_argument(
        "--database", default=os.getenv("INFERSCALE_DATABASE_URL", "sqlite:///inferscale.db")
    )
    p.add_argument("--output", default="results/tuning.json")
    p = subs.add_parser("neuron-compile")
    p.add_argument("config")
    p.add_argument("--output", required=True)
    p = subs.add_parser("token-workload")
    p.add_argument("model")
    p.add_argument("--revision", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--output", default="token-workload.json")
    args = parser.parse_args()
    if args.command == "neuron-compile":
        from .neuron import NeuronCompileConfig, compile_model

        print(
            json.dumps(
                compile_model(
                    NeuronCompileConfig.model_validate(json.loads(Path(args.config).read_text())),
                    args.output,
                ),
                indent=2,
            )
        )
        return
    if args.command == "migrate":
        from .operations import migrate

        print(json.dumps(migrate(args.database)))
        return
    if args.command == "worker":
        from .jobs import JobQueue, work_once, worker_loop

        store = Store(args.database)
        try:
            queue = JobQueue(store)
            asyncio.run(
                work_once(queue, args.executor) if args.once else worker_loop(queue, args.executor)
            )
        finally:
            store.close()
        return
    if args.command == "autoscale":
        from .operations import AutoscaleConfig, autoscale

        asyncio.run(
            autoscale(
                AutoscaleConfig.model_validate(json.loads(Path(args.config).read_text())), args.once
            )
        )
        return
    if args.command == "tune":
        from .tuning import tune

        asyncio.run(tune(args.profiles, args.benchmark, args.database, args.output))
        return
    if args.command == "token-workload":
        from .operations import token_workload

        token_workload(args.model, args.revision, args.lengths, args.output)
        return
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "inferscale.api:app", host=args.host, port=args.port, workers=1, limit_concurrency=256
        )
        return
    if args.command in {"benchmark", "sweep"}:
        raise SystemExit(asyncio.run(execute(args)))
    policy = (
        GatePolicy.model_validate(json.loads(Path(args.policy).read_text()))
        if args.policy
        else GatePolicy()
    )
    if args.command == "gate":
        result = gate(
            json.loads(Path(args.candidate).read_text()),
            json.loads(Path(args.baseline).read_text()) if args.baseline else None,
            policy,
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    result = pareto(json.loads(Path(args.results).read_text()), policy)
    print(json.dumps({"pareto_experiment_ids": result}, indent=2))


if __name__ == "__main__":
    main()
