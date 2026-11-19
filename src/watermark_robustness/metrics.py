"""Record-level removal metrics and descriptive matched-control comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .corpora import load_curation_inputs
from .storage import experiment_manifest, file_hash, write_artifacts


def removal_metrics(train: pd.DataFrame, removed_ids: list[str]) -> dict:
    if train.empty or not train.record_id.is_unique:
        raise ValueError("Metrics require nonempty training data with unique record IDs")
    removed = set(removed_ids)
    if len(removed) != len(removed_ids) or not removed <= set(train.record_id):
        raise ValueError("Removal IDs must be unique members of the training corpus")
    flags = train.get("is_watermarked", pd.Series(False, index=train.index))
    if not pd.api.types.is_bool_dtype(flags) or flags.isna().any():
        raise ValueError("Watermark membership must be boolean and non-null")
    watermarks = set(train.loc[flags, "record_id"])
    n, r, w = len(train), len(removed), len(watermarks)
    removed_watermarks = len(removed & watermarks)
    removed_clean = r - removed_watermarks
    return {
        "training_rows": n,
        "removed_rows": r,
        "retained_rows": n - r,
        "watermark_rows": w,
        "clean_rows": n - w,
        "removed_watermark_rows": removed_watermarks,
        "surviving_watermark_rows": w - removed_watermarks,
        "removed_clean_rows": removed_clean,
        "removal_rate": r / n,
        "watermark_removal_recall": removed_watermarks / w if w else None,
        "watermark_survival_rate": (w - removed_watermarks) / w if w else None,
        "watermark_fraction_among_removed": removed_watermarks / r if r else None,
        "clean_removal_rate": removed_clean / (n - w) if n > w else None,
    }


def compare_random_controls(metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for _, treatment in metrics.loc[metrics.control == "treatment"].iterrows():
        controls = metrics.loc[
            (metrics.corpus == treatment.corpus)
            & (metrics.method == treatment.method)
            & (metrics.requested_budget == treatment.requested_budget)
            & metrics.control.str.startswith("random_")
        ]
        if (
            controls.empty
            or not controls.control_seed.is_unique
            or not (controls.removed_rows == treatment.removed_rows).all()
        ):
            raise ValueError(
                "Each treatment needs unique seeded controls with matched actual counts"
            )
        row = {"condition_id": treatment.condition_id, "random_replicates": len(controls)}
        for metric in (
            "removed_watermark_rows",
            "watermark_removal_recall",
            "watermark_survival_rate",
            "clean_removal_rate",
        ):
            values = pd.to_numeric(controls[metric], errors="raise")
            mean = values.mean()
            row[f"random_{metric}_mean"] = mean
            row[f"random_{metric}_std"] = values.std(ddof=1)
            row[f"treatment_minus_random_{metric}"] = (
                treatment[metric] - mean if pd.notna(treatment[metric]) else None
            )
        comparisons.append(row)
    return pd.DataFrame(comparisons)


def prepare_removal_metrics(config: ExperimentConfig, output_dir: Path | None = None) -> Path:
    output = output_dir or config.metrics_dir
    if output is None:
        raise ValueError("Set paths.metrics_dir or pass --output-dir")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    corpora, curation_manifest, conditions, removals = load_curation_inputs(config)
    selections = removals.groupby("condition_id").record_id.agg(list).to_dict()
    rows = []
    for condition in conditions.to_dict("records"):
        values = removal_metrics(
            corpora[condition["corpus"]], selections.get(condition["condition_id"], [])
        )
        if values["removed_rows"] != condition["actual_removals"]:
            raise ValueError("Metric counts differ from the saved curation selection")
        rows.append({**condition, **values})
    metrics = pd.DataFrame(rows)
    comparisons = compare_random_controls(metrics)
    manifest = experiment_manifest(config)
    manifest.update(
        {
            "kind": "removal_metrics",
            "sources": curation_manifest["sources"],
            "curation_dir": str(config.curation_dir),
            "curation_manifest_sha256": file_hash(config.curation_dir / "manifest.json"),
            "data": {
                "conditions": len(metrics),
                "matched_comparisons": len(comparisons),
                "undefined_rates": "null when the denominator is zero",
                "random_std": "sample standard deviation; null for one replicate",
                "inference": "descriptive removal metrics only; no hypothesis tests",
            },
        }
    )
    return write_artifacts(
        {"removal_metrics": metrics, "random_comparisons": comparisons}, manifest, output
    )
