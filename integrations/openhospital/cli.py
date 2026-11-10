"""Command-line access to the same public Python experiment flow."""
import argparse
import asyncio
from dataclasses import replace
import json

from mematlas.evaluation import ExperimentConfig, ExperimentRunner
from .services import OpenHospitalServices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment JSON, YAML or TOML")
    parser.add_argument("--dry-run", action="store_true", help="Preflight without model calls")
    parser.add_argument("--resume-from", help="Original experiment directory")
    parser.add_argument("--output-dir", help="New output directory when resuming")
    args = parser.parse_args(argv)
    if args.resume_from:
        if args.config or not args.output_dir:
            parser.error("Resume requires --resume-from and a new --output-dir; configuration comes from the original run")
        config = replace(ExperimentConfig.load(args.resume_from), output_dir=args.output_dir)
    else:
        if not args.config or args.output_dir:
            parser.error("Provide --config; set output_dir in that configuration")
        config = ExperimentConfig.load(args.config)
    runner = ExperimentRunner()
    if args.dry_run:
        report = runner.check(config)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["ready"] else 1

    async def run():
        async with OpenHospitalServices(config):
            return await runner.evaluate(config, resume_from=args.resume_from)
    results = asyncio.run(run())
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2, default=str))
    return 1 if any(result.failures for result in results) else 0
