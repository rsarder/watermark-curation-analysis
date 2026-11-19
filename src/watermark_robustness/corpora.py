"""Read the three training corpora and verify their common clean source."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import text_digest
from .protocol import canonical_hash, require_protocol
from .storage import file_hash, read_manifest, read_prepared

CORPORA = ("clean", "unrelated_phrase", "natural_sentence")


def load_training_corpora(config: ExperimentConfig) -> tuple[dict[str, pd.DataFrame], dict]:
    if config.watermark_dir is None:
        raise ValueError("paths.watermark_dir is required")
    clean_dir = config.output_dir.resolve()
    watermark_dir = config.watermark_dir.resolve()
    clean_manifest = read_manifest(clean_dir)
    watermark_manifest = read_manifest(watermark_dir)
    if watermark_manifest.get("kind") != "watermarks":
        raise ValueError("Expected watermarked training artifacts")
    for manifest in (clean_manifest, watermark_manifest):
        require_protocol(manifest, config)
        if manifest["config"]["data"] != asdict(config.data):
            raise ValueError("Data configuration differs from source artifacts")
    clean_hash = file_hash(clean_dir / "manifest.json")
    if watermark_manifest["source_manifest_sha256"] != clean_hash:
        raise ValueError("Watermark artifacts refer to a different clean source")
    if watermark_manifest["watermark_settings"] != asdict(config.watermark):
        raise ValueError("Watermark settings differ from source artifacts")
    corpora = {"clean": read_prepared(clean_dir).splits["train"]}
    for name in CORPORA[1:]:
        corpora[name] = pd.read_parquet(watermark_dir / name / "train.parquet")
    reference_ids = set(corpora["clean"].record_id)
    for name, frame in corpora.items():
        if not frame.record_id.is_unique or set(frame.record_id) != reference_ids:
            raise ValueError(f"Training source IDs differ in {name}")
        if set(frame.source_split) != {"train"}:
            raise ValueError("Only official training records may enter curation")
        if not frame.text_sha256.equals(frame.text.map(text_digest)):
            raise ValueError(f"Text fingerprint mismatch in {name}")
        corpora[name] = frame.sort_values("record_id").reset_index(drop=True)
    sources = {
        "clean_dir": str(clean_dir),
        "watermark_dir": str(watermark_dir),
        "clean_manifest_sha256": clean_hash,
        "watermark_manifest_sha256": file_hash(watermark_dir / "manifest.json"),
    }
    return corpora, sources


def training_path(sources: dict, corpus: str) -> Path:
    if corpus not in CORPORA:
        raise ValueError(f"Unknown training corpus: {corpus}")
    if corpus == "clean":
        return Path(sources["clean_dir"]) / "train.parquet"
    return Path(sources["watermark_dir"]) / corpus / "train.parquet"


def load_curation_inputs(
    config: ExperimentConfig,
) -> tuple[dict[str, pd.DataFrame], dict, pd.DataFrame, pd.DataFrame]:
    """Audit curation selections once before metrics or classifier training."""
    if config.curation_dir is None:
        raise ValueError("paths.curation_dir is required")
    corpora, sources = load_training_corpora(config)
    manifest = read_manifest(config.curation_dir)
    require_protocol(manifest, config)
    if canonical_hash(manifest.get("curation_settings")) != canonical_hash(asdict(config.curation)):
        raise ValueError("Curation settings differ from the requested protocol")
    if manifest.get("kind") != "curation" or manifest["sources"] != sources:
        raise ValueError("Curation refers to a different training source")
    conditions = pd.read_parquet(config.curation_dir / "conditions.parquet")
    removals = pd.read_parquet(config.curation_dir / "removals.parquet")
    if (
        conditions.empty
        or not conditions.condition_id.is_unique
        or not set(conditions.corpus) <= set(corpora)
    ):
        raise ValueError("Invalid curation condition identities")
    if (
        not set(removals.condition_id) <= set(conditions.condition_id)
        or removals.duplicated(["condition_id", "record_id"]).any()
    ):
        raise ValueError("Unknown conditions or repeated removal IDs")
    counts = removals.groupby("condition_id").size()
    if (
        conditions.condition_id.map(counts).fillna(0).tolist()
        != conditions.actual_removals.tolist()
    ):
        raise ValueError("Removal counts differ from curation conditions")
    for corpus, frame in corpora.items():
        condition_ids = conditions.loc[conditions.corpus == corpus, "condition_id"]
        if not set(removals.loc[removals.condition_id.isin(condition_ids), "record_id"]) <= set(
            frame.record_id
        ):
            raise ValueError("Removed IDs are not members of their training corpus")
    return corpora, manifest, conditions, removals
