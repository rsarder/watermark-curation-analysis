"""Execute paired-sign-v1 on saved models and the already frozen common panel."""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .corpora import load_curation_inputs
from .data import text_digest
from .identity import input_identities, load_tokenizer
from .protocol import LOCK, ROOT, canonical_hash, require_frozen_source, require_protocol
from .storage import (
    experiment_manifest,
    file_hash,
    read_manifest,
    read_prepared,
    require_prepared_protocol,
    write_artifacts,
)
from .training import evaluate_model, load_checkpoint
from .verification_stats import decision_status, holm_correction, paired_effects

VARIANTS = ("unrelated_phrase", "natural_sentence")
POOLS = ("calibration_probe", "evaluation_probe")
PAIR_COLUMNS = ["record_id", "unit_id", "label", "w_clean", "w_triggered", "c_clean", "c_triggered"]


def model_key(condition: str, seed: int) -> str:
    return f"{condition}/seed_{seed}"


def clean_reference(condition: str) -> str:
    corpus, suffix = condition.split("/", 1)
    if corpus not in VARIANTS:
        raise ValueError("Expected a watermarked condition")
    return f"clean/{suffix}"


def hypothesis_family(config: ExperimentConfig, conditions: pd.DataFrame) -> list[str]:
    """Use all 150 NAMED tests, even when retained datasets or fits coincide."""
    settings = config.plan["verification"]
    if (
        settings.get("plan_version") != "paired-sign-v1"
        or settings.get("alpha") != 0.05
        or settings.get("correction") != "holm"
        or settings.get("family_size") != 150
        or tuple(config.classifier.training_seeds) != (42, 43, 44)
    ):
        raise ValueError("Verification requires the frozen full-experiment family")
    suffixes = {"uncurated"}
    for method in ("semantic_deduplication", "embedding_knn_outlier"):
        for budget in ("0.01", "0.05", "0.1"):
            for control in ("treatment", "random_42", "random_43", "random_44"):
                suffixes.add(f"{method}/{budget}/{control}")
    expected = {f"{corpus}/{suffix}" for corpus in ("clean", *VARIANTS) for suffix in suffixes}
    if not conditions.condition_id.is_unique or set(conditions.condition_id) != expected:
        raise ValueError("Conditions differ from the frozen 75-condition design")
    return sorted(
        model_key(condition, seed)
        for condition in expected
        if condition.split("/", 1)[0] in VARIANTS
        for seed in config.classifier.training_seeds
    )


def load_context(config: ExperimentConfig, pool: str):
    if pool not in POOLS:
        raise ValueError("Choose calibration_probe or evaluation_probe explicitly")
    require_prepared_protocol(config.output_dir, config)
    _, curated, conditions, _ = load_curation_inputs(config)
    data = read_prepared(config.output_dir)
    units = pd.read_parquet(config.watermark_dir / "verification_units.parquet")
    if (
        not units.record_id.is_unique
        or not units.unit_id.is_unique
        or units.isna().any().any()
        or not units.eligible.equals(
            units.unrelated_phrase_visible & units.natural_sentence_visible
        )
    ):
        raise ValueError("Invalid frozen verification units")
    panel = (
        units.loc[(units.pool == pool) & units.eligible]
        .sort_values("record_id")
        .reset_index(drop=True)
    )
    if panel.empty or set(panel.label) != {1, 2, 3}:
        raise ValueError("The common panel must cover all original non-target classes")
    ids = panel.record_id.tolist()
    views = {"clean": data.splits[pool].set_index("record_id").loc[ids].reset_index()}
    tokenizer = load_tokenizer(config.classifier, config.cache_dir.parent / "models")
    for variant in VARIANTS:
        frame = pd.read_parquet(config.watermark_dir / variant / f"{pool}.parquet")
        if not frame.record_id.is_unique:
            raise ValueError("Repeated probe IDs")
        frame = frame.set_index("record_id").loc[ids].reset_index()
        if (
            not frame.original_text.equals(views["clean"].text)
            or not frame.original_label.equals(views["clean"].label)
            or not frame.target_label.eq(0).all()
            or not frame.text.equals(frame.original_text + " " + config.watermark.triggers[variant])
        ):
            raise ValueError("Triggered probes differ from frozen paired views")
        views[variant] = frame
    for view, frame in views.items():
        expected = panel.unit_id if view == "clean" else panel[f"{view}_input_sha256"]
        if (
            frame.record_id.tolist() != ids
            or frame.label.tolist() != panel.label.tolist()
            or not frame.text_sha256.equals(frame.text.map(text_digest))
            or input_identities(frame.text.tolist(), tokenizer, config.classifier.max_length)
            != expected.tolist()
        ):
            raise ValueError("Probe text, labels, order, or classifier input identities differ")
    return panel, views, curated, conditions


def model_registry(config, conditions, curated, source_hash):
    runs = {}
    if config.training_dir is None:
        raise ValueError("paths.training_dir is required")
    for condition in sorted(conditions.condition_id):
        for seed in config.classifier.training_seeds:
            directory = config.training_dir / condition / f"seed_{seed}"
            if not (directory / "manifest.json").exists():
                continue
            manifest = read_manifest(directory)
            require_protocol(manifest, config)
            if (
                manifest.get("kind") != "classifier_training"
                or manifest.get("condition_id") != condition
                or manifest.get("training_seed") != seed
                or manifest.get("reviewed_source_sha256") != source_hash
                or canonical_hash(manifest.get("classifier_settings"))
                != canonical_hash(asdict(config.classifier))
                or manifest.get("sources") != curated["sources"]
                or manifest.get("curation_manifest_sha256")
                != file_hash(config.curation_dir / "manifest.json")
            ):
                raise ValueError(f"Model provenance/source revision mismatch: {directory}")
            expected_checkpoint = {
                f"checkpoint/{name}"
                for name in (
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            }
            if not expected_checkpoint <= set(manifest["files"]):
                raise ValueError("Incomplete classifier checkpoint")
            details = manifest["data"]
            if (
                details["smoke_run"]
                or not details["checkpoint_reload_verified"]
                or not details["classifier_head_updated"]
            ):
                raise ValueError("Verification requires completed full classifier runs")
            history = pd.read_parquet(directory / "history.parquet")
            if history.epoch.tolist() != list(range(1, config.classifier.epochs + 1)):
                raise ValueError("Incomplete training history")
            best = history.sort_values(
                ["validation_accuracy", "validation_loss", "epoch"], ascending=[False, True, True]
            ).iloc[0]
            if (
                best.epoch != details["best_epoch"]
                or best.validation_accuracy != details["best_validation_accuracy"]
                or best.validation_loss != details["best_validation_loss"]
            ):
                raise ValueError("Saved model violates clean-validation checkpoint selection")
            runs[model_key(condition, seed)] = {
                "directory": directory,
                "condition": condition,
                "seed": seed,
                "manifest_sha256": file_hash(directory / "manifest.json"),
                "best_epoch": int(best.epoch),
                "validation_accuracy": float(best.validation_accuracy),
                "validation_loss": float(best.validation_loss),
            }
    if not runs:
        raise ValueError("No completed full models exist; no predictions were run")
    for run in runs.values():
        if run["condition"].split("/", 1)[0] in VARIANTS:
            if model_key(clean_reference(run["condition"]), run["seed"]) not in runs:
                raise ValueError("Missing same-seed, policy-matched clean-trained control")
    return runs


def predict_run(directory, views, settings):
    """Load one model at a time; call only on supplied probe views, never official test."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model, tokenizer = load_checkpoint(directory / "checkpoint")
    model.to(settings.device)
    try:
        return {
            name: evaluate_model(model, tokenizer, frame, settings)[0]
            for name, frame in views.items()
        }
    finally:
        del model
        if settings.device == "cuda":
            torch.cuda.empty_cache()


def align_predictions(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Align by stable IDs, reject missing/extra records, and preserve frozen labels."""
    if (
        not {"record_id", "label", "predicted_label"} <= set(predictions.columns)
        or not predictions.record_id.is_unique
        or set(predictions.record_id) != set(panel.record_id)
    ):
        raise ValueError("Missing, extra, or repeated prediction IDs")
    aligned = predictions.set_index("record_id").loc[panel.record_id].reset_index()
    if (
        aligned.label.tolist() != panel.label.tolist()
        or aligned.isna().any().any()
        or not pd.api.types.is_integer_dtype(aligned.predicted_label)
        or not set(aligned.predicted_label) <= {0, 1, 2, 3}
    ):
        raise ValueError("Prediction labels do not match the frozen panel")
    if "unit_id" in aligned and aligned.unit_id.tolist() != panel.unit_id.tolist():
        raise ValueError("Mismatched verification cluster identities")
    aligned["unit_id"] = panel.unit_id.tolist()
    return aligned


def pair_predictions(predictions, panel, watermarked, clean, variant):
    pairs = panel[["record_id", "unit_id", "label"]].copy()
    for name, key, view in (
        ("w_clean", watermarked, "clean"),
        ("w_triggered", watermarked, variant),
        ("c_clean", clean, "clean"),
        ("c_triggered", clean, variant),
    ):
        selected = predictions.loc[(predictions.model_key == key) & (predictions.view == view)]
        aligned = align_predictions(selected, panel)
        pairs[name] = aligned.predicted_label.tolist()
    return pairs


def summarize_predictions(predictions, panel, runs, pool, family, final=False):
    effects, pair_tables, responses, controls = [], [], [], []
    for key, run in runs.items():
        corpus = run["condition"].split("/", 1)[0]
        for variant in VARIANTS if corpus == "clean" else (corpus,):
            self_pairs = pair_predictions(predictions, panel, key, key, variant)
            for row in paired_effects(self_pairs).to_dict("records"):
                responses.append(
                    {
                        "model_key": key,
                        "variant": variant,
                        "scope": row["scope"],
                        "units": row["units"],
                        "clean_target_rate": row["w_clean_target_rate"],
                        "triggered_target_rate": row["w_triggered_target_rate"],
                        "trigger_effect": row["mean_a_w"],
                    }
                )
                if corpus == "clean" and pool == "calibration_probe":
                    controls.append(
                        {
                            "control_kind": "identical_clean_self",
                            "variant": variant,
                            "model_w": key,
                            "model_c": key,
                            **row,
                        }
                    )
        if corpus in VARIANTS:
            reference = model_key(clean_reference(run["condition"]), run["seed"])
            pairs = pair_predictions(predictions, panel, key, reference, corpus)
            pair_tables.append(
                pairs.assign(hypothesis_id=key, reference_model=reference, variant=corpus)
            )
            effects.extend(
                {"hypothesis_id": key, "reference_model": reference, "variant": corpus, **row}
                for row in paired_effects(pairs).to_dict("records")
            )
    if pool == "calibration_probe":
        policies = sorted(
            {run["condition"] for run in runs.values() if run["condition"].startswith("clean/")}
        )
        for policy in policies:
            for first, second in combinations((42, 43, 44), 2):
                w, c = model_key(policy, first), model_key(policy, second)
                if w in runs and c in runs:
                    for variant in VARIANTS:
                        controls.extend(
                            {
                                "control_kind": "clean_seed_pair",
                                "variant": variant,
                                "model_w": w,
                                "model_c": c,
                                **row,
                            }
                            for row in paired_effects(
                                pair_predictions(predictions, panel, w, c, variant)
                            ).to_dict("records")
                        )
    effect_table = pd.DataFrame(effects)
    overall = (
        effect_table.loc[effect_table.scope == "overall"]
        if effects
        else pd.DataFrame(columns=["hypothesis_id", "p_value"])
    )
    corrected = holm_correction(overall, family)
    complete = bool(corrected.observed.all())
    if final and (pool != "evaluation_probe" or not complete):
        raise ValueError("Final decisions require the complete evaluation hypothesis family")
    corrected["decision"] = [
        decision_status(
            pool,
            complete,
            Fraction(int(row.adjusted_p_value_numerator), int(row.adjusted_p_value_denominator)),
            0.05,
            final=final,
        )
        for row in corrected.itertuples()
    ]
    return {
        "predictions": predictions,
        "paired_predictions": pd.concat(pair_tables, ignore_index=True)
        if pair_tables
        else pd.DataFrame(columns=[*PAIR_COLUMNS, "hypothesis_id", "reference_model", "variant"]),
        "effects": effect_table
        if effects
        else pd.DataFrame(columns=["hypothesis_id", "scope", "p_value"]),
        "holm": corrected,
        "trigger_responses": pd.DataFrame(responses),
        "negative_controls": pd.DataFrame(controls)
        if controls
        else pd.DataFrame(
            columns=["control_kind", "variant", "model_w", "model_c", "scope", "p_value"]
        ),
    }


def run_verification(
    config: ExperimentConfig, output_dir: Path, *, pool="calibration_probe", final=False
) -> Path:
    """Inference is opt-in via verify; no training, model selection, or official-test inference."""
    import json

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Verification output already exists: {output_dir}")
    if final and pool != "evaluation_probe":
        raise ValueError("Calibration cannot produce confirmatory rejection decisions")
    source_hash = require_frozen_source()
    plan_path = ROOT / "docs/VERIFICATION_PLAN.md"
    original = json.loads(LOCK.read_text(encoding="utf-8"))
    if file_hash(plan_path) != original["files"]["docs/VERIFICATION_PLAN.md"]:
        raise ValueError("Frozen statistical plan changed")
    panel, views, curated, conditions = load_context(config, pool)
    family = hypothesis_family(config, conditions)
    runs = model_registry(config, conditions, curated, source_hash)
    if final and not set(family) <= set(runs):
        raise ValueError("Final evaluation requires all 150 watermarked models and their controls")
    start = time.perf_counter()
    tables = []
    for key, run in runs.items():
        corpus = run["condition"].split("/", 1)[0]
        selected_views = (
            views if corpus == "clean" else {"clean": views["clean"], corpus: views[corpus]}
        )
        predictions = predict_run(run["directory"], selected_views, config.classifier)
        if set(predictions) != set(selected_views):
            raise ValueError("Prediction views differ from the requested frozen pairs")
        tables.extend(
            align_predictions(frame, panel).assign(model_key=key, view=view)
            for view, frame in predictions.items()
        )
    frames = summarize_predictions(
        pd.concat(tables, ignore_index=True), panel, runs, pool, family, final=final
    )
    observed = int(frames["holm"].observed.sum())
    manifest = experiment_manifest(config)
    manifest.update(
        {
            "kind": "verification",
            "reviewed_source_sha256": source_hash,
            "verification_plan_sha256": file_hash(plan_path),
            "sources": curated["sources"],
            "curation_manifest_sha256": file_hash(config.curation_dir / "manifest.json"),
            "models": {
                key: {**run, "directory": str(run["directory"])} for key, run in runs.items()
            },
            "data": {
                "pool": pool,
                "panel_units": len(panel),
                "panel_sha256": canonical_hash(panel.to_json(orient="records")),
                "models": len(runs),
                "observed_hypotheses": observed,
                "family_size": len(family),
                "unrun_p_value": 1,
                "correction": "holm",
                "alpha": 0.05,
                "confirmatory_decisions_made": final,
                "duration_seconds": time.perf_counter() - start,
                "class_specific_tests": "descriptive; not additional confirmatory tests",
                "null": "within each non-target class P(D>0 | D!=0) <= 1/2",
                "alternative": "excess positive signs; violation in at least one class",
                "ties": "omit zero differences; report count; all ties gives p=1",
                "dependence": (
                    "assumes independent article units; exact-input uniqueness "
                    "does not prove story independence"
                ),
                "official_test_predictions": False,
                "model_selection": "previously frozen clean-validation selection only",
            },
        }
    )
    return write_artifacts(frames, manifest, output_dir)
