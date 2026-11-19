"""Command line entry point for data preparation and artifact validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .audit import audit_experiment
from .config import load_config
from .curation import prepare_curation
from .data import load_ag_news
from .embeddings import prepare_embeddings
from .identity import load_tokenizer
from .metrics import prepare_removal_metrics
from .protocol import freeze_source
from .storage import read_manifest, read_prepared, write_prepared
from .training import train_classifier
from .verification import run_verification
from .watermark import prepare_watermarks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare clean AG News partitions only")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, help="Override output path, relative to cwd")
    watermark = commands.add_parser("watermark", help="Build both paired dirty-label variants")
    watermark.add_argument("--config", type=Path, required=True)
    watermark.add_argument(
        "--input-dir", type=Path, help="Override prepared clean source directory"
    )
    watermark.add_argument("--output-dir", type=Path, help="Override watermarked output directory")
    embed = commands.add_parser("embed", help="Embed clean and watermarked training text")
    embed.add_argument("--config", type=Path, required=True)
    embed.add_argument("--output-dir", type=Path)
    curate = commands.add_parser("curate", help="Create curation and matched random selections")
    curate.add_argument("--config", type=Path, required=True)
    curate.add_argument("--embedding-dir", type=Path)
    curate.add_argument("--output-dir", type=Path)
    metrics = commands.add_parser("metrics", help="Measure record removal and matched controls")
    metrics.add_argument("--config", type=Path, required=True)
    metrics.add_argument("--output-dir", type=Path)
    train = commands.add_parser("train", help="Train one DistilBERT condition and seed")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--condition", default="clean/uncurated")
    train.add_argument("--seed", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--resume", action="store_true", help="Resume this run's last atomic state")
    freeze = commands.add_parser("freeze-source", help="Append a reviewed source/config revision")
    freeze.add_argument("--revision", help="New immutable implementation revision name")
    verify = commands.add_parser(
        "verify", help="Verify saved models using the frozen paired-sign plan"
    )
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument(
        "--pool", choices=("calibration_probe", "evaluation_probe"), default="calibration_probe"
    )
    verify.add_argument(
        "--final",
        action="store_true",
        help="Make the final evaluation decision; requires the complete family",
    )
    audit = commands.add_parser("audit", help="Read-only replay of corrected experiment artifacts")
    audit.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="Validate and summarize saved partitions")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-source":
            print(json.dumps({"archive": str(freeze_source(args.revision))}))
        elif args.command == "verify":
            output = run_verification(
                load_config(args.config), args.output_dir, pool=args.pool, final=args.final
            )
            print(
                json.dumps(
                    {"output_dir": str(output), "data": read_manifest(output)["data"]}, indent=2
                )
            )
        elif args.command == "audit":
            print(json.dumps(audit_experiment(load_config(args.config)), indent=2))
        elif args.command == "prepare":
            config = load_config(args.config)
            if args.output_dir is not None:
                config = replace(config, output_dir=args.output_dir.resolve())
            if config.output_dir.exists():
                raise FileExistsError(f"Output already exists: {config.output_dir}")
            tokenizer = load_tokenizer(config.classifier, config.cache_dir.parent / "models")
            data = load_ag_news(
                config.data,
                cache_dir=str(config.cache_dir),
                tokenizer=tokenizer,
                classifier=config.classifier,
            )
            output = write_prepared(data, config)
            print(json.dumps({"output_dir": str(output), "data": data.metadata}, indent=2))
        elif args.command == "watermark":
            output = prepare_watermarks(load_config(args.config), args.input_dir, args.output_dir)
            manifest = read_manifest(output)
            print(json.dumps({"output_dir": str(output), "data": manifest["data"]}, indent=2))
        elif args.command in {"embed", "curate"}:
            config = load_config(args.config)
            output = (
                prepare_embeddings(config, args.output_dir)
                if args.command == "embed"
                else prepare_curation(config, args.embedding_dir, args.output_dir)
            )
            print(
                json.dumps(
                    {"output_dir": str(output), "data": read_manifest(output)["data"]}, indent=2
                )
            )
        elif args.command in {"metrics", "train"}:
            config = load_config(args.config)
            output = (
                prepare_removal_metrics(config, args.output_dir)
                if args.command == "metrics"
                else train_classifier(
                    config, args.condition, args.seed, args.output_dir, resume=args.resume
                )
            )
            print(
                json.dumps(
                    {"output_dir": str(output), "data": read_manifest(output)["data"]}, indent=2
                )
            )
        else:
            manifest = read_manifest(args.directory)
            metadata = (
                manifest["data"]
                if manifest.get("kind")
                in {
                    "watermarks",
                    "embeddings",
                    "curation",
                    "removal_metrics",
                    "classifier_training",
                    "verification",
                }
                else read_prepared(args.directory).metadata
            )
            print(json.dumps(metadata, indent=2))
    except ModuleNotFoundError as error:
        parser.exit(
            1,
            f"error: {error}. Install with pip install -e .[curation,training]\n",
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0
