"""Parquet artifacts with integrity checks and a reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd

from . import __version__
from .config import DataConfig, ExperimentConfig
from .data import (
    SPLIT_NAMES,
    PreparedData,
    source_fingerprint,
    split_statistics,
    validate_partitions,
)
from .identity import tokenizer_contract
from .protocol import protocol_hash, require_protocol


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def staged_directory(output: Path):
    """Publish only successful complete runs, inheriting workspace permissions."""
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # mkdir inherits workspace permissions on Windows. tempfile.mkdtemp uses
    # owner-only permissions, which would survive publication by rename.
    staging = output.parent / f".{output.name}-{uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_manifest(directory: Path, manifest: dict) -> None:
    files = {
        path.relative_to(directory).as_posix(): {"sha256": file_hash(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }
    (directory / "manifest.json").write_text(
        json.dumps({**manifest, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_artifacts(frames: dict[str, pd.DataFrame], manifest: dict, output: Path) -> Path:
    """Publish a complete new directory; never overwrite an existing run."""
    with staged_directory(output) as staging:
        for name, frame in frames.items():
            path = staging / f"{name}.parquet"
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Artifact names must stay inside the output directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False, engine="pyarrow")
        write_manifest(staging, manifest)
    return output


def experiment_manifest(config: ExperimentConfig) -> dict:
    config_dict = config.to_dict()
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "protocol_version": config.protocol_version,
        "protocol_sha256": protocol_hash(config),
        "package_version": __version__,
        "config": config_dict,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                name: version(name)
                for name in ("datasets", "numpy", "pandas", "pyarrow", "scikit-learn")
            },
        },
    }


def write_prepared(data: PreparedData, config: ExperimentConfig) -> Path:
    validate_partitions(data, config.data)
    if data.identities is None or data.metadata.get("tokenizer_contract") != tokenizer_contract(
        config.classifier
    ):
        raise ValueError("Prepared data lacks the current classifier tokenization audit")
    manifest = experiment_manifest(config)
    manifest["data"] = data.metadata
    return write_artifacts(
        {**data.splits, "excluded": data.excluded, "input_identities": data.identities},
        manifest,
        config.output_dir,
    )


def read_manifest(directory: str | Path) -> dict:
    """Verify stored configuration and Parquet file checksums."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema version")
    canonical = json.dumps(manifest["config"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Configuration checksum mismatch")
    for filename, info in manifest["files"].items():
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("Artifact paths must stay inside the output directory")
        if file_hash(directory / filename) != info["sha256"]:
            raise ValueError(f"Parquet checksum mismatch: {filename}")
    return manifest


def read_prepared(directory: str | Path) -> PreparedData:
    """Read artifacts only after checking their bytes and partition contracts."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    expected_files = {f"{name}.parquet" for name in (*SPLIT_NAMES, "excluded")}
    corrected = manifest.get("protocol_version") == "tokenized-v2"
    if corrected:
        expected_files.add("input_identities.parquet")
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected artifact file set")
    result = PreparedData(
        splits={name: pd.read_parquet(directory / f"{name}.parquet") for name in SPLIT_NAMES},
        excluded=pd.read_parquet(directory / "excluded.parquet"),
        metadata=manifest["data"],
        identities=pd.read_parquet(directory / "input_identities.parquet") if corrected else None,
    )
    config = DataConfig(**manifest["config"]["data"])
    validate_partitions(result, config)
    actual = {name: split_statistics(frame) for name, frame in result.splits.items()}
    if actual != result.metadata["splits"]:
        raise ValueError("Partition statistics mismatch")
    if len(result.excluded) != result.metadata["excluded_rows"]:
        raise ValueError("Exclusion count mismatch")
    if source_fingerprint(result.splits["test"]) != result.metadata["source"]["test_fingerprint"]:
        raise ValueError("Official test fingerprint mismatch")
    return result


def require_prepared_protocol(directory: Path, config: ExperimentConfig) -> None:
    manifest = read_manifest(directory)
    require_protocol(manifest, config)
    if manifest["data"].get("tokenizer_contract") != tokenizer_contract(config.classifier):
        raise ValueError("Prepared data uses different classifier tokenization")
