"""Paired dirty-label training watermarks and held-out clean/triggered probes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, ExperimentConfig, WatermarkConfig
from .data import PreparedData, text_digest, validate_partitions
from .identity import input_identities, load_tokenizer, tokenize
from .storage import (
    experiment_manifest,
    file_hash,
    read_prepared,
    require_prepared_protocol,
    write_artifacts,
)

PROBE_POOLS = ("calibration_probe", "evaluation_probe")


@dataclass
class WatermarkedData:
    frames: dict[str, pd.DataFrame]
    metadata: dict


def select_source_ids(
    train: pd.DataFrame, settings: WatermarkConfig, target: int, seed: int
) -> list[str]:
    """Sample IDs once, independently of dataframe order and trigger wording."""
    if not train.record_id.is_unique or set(train.source_split) != {"train"}:
        raise ValueError("Watermark sources must be unique official training records")
    count = int(Decimal(str(settings.injection_rate)) * len(train))
    eligible = sorted(train.loc[train.label != target, "record_id"])
    if count < 1 or count > len(eligible):
        raise ValueError("Injection count must be positive and fit the non-target training pool")
    return sorted(np.random.default_rng(seed).choice(eligible, size=count, replace=False).tolist())


def _with_originals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result["original_text"] = result["text"]
    result["original_label"] = result["label"]
    result["original_text_sha256"] = result["text_sha256"]
    return result


def build_watermarks(
    data: PreparedData, data_config: DataConfig, settings: WatermarkConfig
) -> WatermarkedData:
    validate_partitions(data, data_config)
    train = data.splits["train"]
    selected_ids = select_source_ids(train, settings, data_config.target_label, data_config.seed)
    selected = train.record_id.isin(selected_ids)
    frames = {
        "selected_sources": train.loc[selected].sort_values("record_id").reset_index(drop=True)
    }
    for variant, trigger in settings.triggers.items():
        # An existing trigger would make the clean/triggered comparison ambiguous.
        if any(
            frame.text.str.contains(trigger, case=False, regex=False).any()
            for frame in data.splits.values()
        ):
            raise ValueError(f"Trigger for {variant} already occurs in clean source text")
        watermarked = _with_originals(train)
        watermarked["is_watermarked"] = selected.to_numpy()
        watermarked.loc[selected, "text"] = watermarked.loc[selected, "text"] + " " + trigger
        watermarked.loc[selected, "label"] = data_config.target_label
        watermarked["text_sha256"] = watermarked.text.map(text_digest)
        frames[f"{variant}/train"] = watermarked
        for pool in PROBE_POOLS:
            probe = _with_originals(data.splits[pool])
            # Probe labels stay original. original_text is the paired clean input.
            probe["text"] = probe["original_text"] + " " + trigger
            probe["text_sha256"] = probe.text.map(text_digest)
            probe["target_label"] = data_config.target_label
            frames[f"{variant}/{pool}"] = probe
    return WatermarkedData(
        frames=frames,
        metadata={
            "training_rows": len(train),
            "injected_rows_per_variant": len(selected_ids),
            "requested_injection_rate": settings.injection_rate,
            "actual_injection_rate": len(selected_ids) / len(train),
            "selection_seed": data_config.seed,
            "target_label": data_config.target_label,
            "triggers": settings.triggers,
            "probe_rows_per_variant": {pool: len(data.splits[pool]) for pool in PROBE_POOLS},
            "probe_clean_text_column": "original_text",
            "probe_triggered_text_column": "text",
            "source_test_fingerprint": data.metadata["source"]["test_fingerprint"],
        },
    )


def prepare_watermarks(
    config: ExperimentConfig, source_dir: Path | None = None, output_dir: Path | None = None
) -> Path:
    source_dir = (source_dir or config.output_dir).resolve()
    output_dir = output_dir or config.watermark_dir
    if output_dir is None:
        raise ValueError("Set paths.watermark_dir or pass --output-dir")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    source_manifest = source_dir / "manifest.json"
    require_prepared_protocol(source_dir, config)
    # read_prepared verifies source file hashes before any injection.
    source = read_prepared(source_dir)
    source_config = json.loads(source_manifest.read_text(encoding="utf-8"))["config"]["data"]
    if source_config != asdict(config.data):
        raise ValueError("Configuration data settings differ from the prepared source")
    result = build_watermarks(source, config.data, config.watermark)
    tokenizer = load_tokenizer(config.classifier, config.cache_dir.parent / "models")
    units = verification_units(source, result, tokenizer, config.classifier.max_length)
    result.frames["verification_units"] = units
    result.metadata["verification_panel_rows"] = (
        units.loc[units.eligible].groupby("pool").size().to_dict()
    )
    manifest = experiment_manifest(config)
    manifest.update(
        {
            "kind": "watermarks",
            "source_dir": str(source_dir),
            "source_manifest_sha256": file_hash(source_manifest),
            "watermark_settings": asdict(config.watermark),
            "data": result.metadata,
        }
    )
    return write_artifacts(result.frames, manifest, output_dir)


def verification_units(
    source: PreparedData, result: WatermarkedData, tokenizer, max_length: int
) -> pd.DataFrame:
    """Audit all actual views without model predictions; freeze a common trigger panel."""
    views = {}
    for name, frame in source.splits.items():
        views[name] = set(input_identities(frame.text.tolist(), tokenizer, max_length))
    rows = []
    for pool in PROBE_POOLS:
        frame = source.splits[pool]
        panel = frame[["record_id", "label"]].copy()
        panel["pool"] = pool
        panel["unit_id"] = input_identities(frame.text.tolist(), tokenizer, max_length)
        panel["eligible"] = True
        for variant, trigger in result.metadata["triggers"].items():
            triggered = result.frames[f"{variant}/{pool}"]
            hashes = input_identities(triggered.text.tolist(), tokenizer, max_length)
            panel[f"{variant}_input_sha256"] = hashes
            views[pool].update(hashes)
            ids = tokenize(triggered.text.tolist(), tokenizer, max_length)["input_ids"]
            trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
            visible = [values[1:-1][-len(trigger_ids) :] == trigger_ids for values in ids]
            panel[f"{variant}_visible"] = visible
            panel["eligible"] &= pd.Series(visible, index=panel.index)
        rows.append(panel)
    for variant in result.metadata["triggers"]:
        frame = result.frames[f"{variant}/train"]
        views["train"].update(input_identities(frame.text.tolist(), tokenizer, max_length))
    names = list(views)
    for pos, left in enumerate(names):
        for right in names[pos + 1 :]:
            if views[left] & views[right]:
                raise ValueError(f"Watermarked classifier input leakage: {left}/{right}")
    units = pd.concat(rows, ignore_index=True)
    if units.unit_id.duplicated().any():
        raise ValueError("Probe verification units are not unique")
    for variant in result.metadata["triggers"]:
        eligible = units.loc[units.eligible, f"{variant}_input_sha256"]
        if eligible.duplicated().any():
            raise ValueError("Repeated triggered input in the common verification panel")
    return units
