"""Pairing and execution regressions use synthetic predictions, never real probes."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from watermark_robustness.config import load_config
from watermark_robustness.curation import removal_conditions
from watermark_robustness.storage import (
    experiment_manifest,
    file_hash,
    read_manifest,
    write_artifacts,
    write_manifest,
)
from watermark_robustness.verification import (
    VARIANTS,
    align_predictions,
    clean_reference,
    hypothesis_family,
    model_key,
    model_registry,
    pair_predictions,
    predict_run,
    run_verification,
    summarize_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


def conditions(config):
    scores = pd.DataFrame(
        {
            "record_id": [f"training-{i}" for i in range(20)],
            "knn_outlier_score": np.arange(20, dtype=float),
            "similarity_to_representative": 1.0,
            "duplicate_eligible": True,
        }
    )
    return pd.concat(
        [removal_conditions(scores, corpus, config.curation)[0] for corpus in ("clean", *VARIANTS)],
        ignore_index=True,
    )


def synthetic_context(config):
    panel = pd.DataFrame(
        {
            "record_id": [f"synthetic-{i}" for i in range(6)],
            "unit_id": [f"synthetic-unit-{i}" for i in range(6)],
            "label": [1, 2, 3, 1, 2, 3],
        }
    )
    views = {
        name: panel[["record_id", "label"]].assign(text=f"synthetic {name}")
        for name in ("clean", *VARIANTS)
    }
    runs = {
        model_key(f"{corpus}/uncurated", 42): {
            "condition": f"{corpus}/uncurated",
            "seed": 42,
            "directory": Path(corpus),
            "manifest_sha256": "synthetic",
            "best_epoch": 1,
            "validation_accuracy": 0.75,
            "validation_loss": 0.5,
        }
        for corpus in ("clean", *VARIANTS)
    }
    return panel, views, runs


class VerificationExecutionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/full.toml")

    def test_frozen_family_and_clean_reference_mapping(self):
        family = hypothesis_family(self.config, conditions(self.config))
        self.assertEqual(len(family), 150)
        self.assertEqual(len(set(family)), 150)
        self.assertEqual(clean_reference("natural_sentence/uncurated"), "clean/uncurated")
        self.assertEqual(
            clean_reference("unrelated_phrase/semantic_deduplication/0.1/random_43"),
            "clean/semantic_deduplication/0.1/random_43",
        )
        with self.assertRaises(ValueError):
            hypothesis_family(self.config, conditions(self.config).iloc[:-1])

    def test_real_prediction_execution_uses_only_synthetic_cpu_views(self):
        import torch
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        from watermark_robustness.identity import load_tokenizer

        panel, views, _ = synthetic_context(self.config)
        tokenizer = load_tokenizer(self.config.classifier, self.config.cache_dir.parent / "models")
        settings = replace(self.config.classifier, device="cpu", threads=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny = DistilBertForSequenceClassification(
                DistilBertConfig(
                    vocab_size=30522,
                    n_layers=1,
                    n_heads=2,
                    dim=16,
                    hidden_dim=32,
                    num_labels=4,
                )
            )
            tiny.save_pretrained(root / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(root / "checkpoint")
            predictions = predict_run(root, views, settings)
            self.assertEqual(set(predictions), {"clean", *VARIANTS})
            for frame in predictions.values():
                self.assertEqual(frame.record_id.tolist(), panel.record_id.tolist())
                self.assertEqual(frame.label.tolist(), panel.label.tolist())
                probabilities = frame[[f"probability_{label}" for label in range(4)]].to_numpy()
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                self.assertTrue(torch.isfinite(torch.tensor(probabilities)).all().item())

    def test_pairing_reorders_by_id_and_rejects_wrong_labels_clusters_and_missing_units(self):
        panel, _, _ = synthetic_context(self.config)
        predictions = panel[["record_id", "label"]].assign(predicted_label=[0, 1, 2, 3, 0, 1])
        aligned = align_predictions(predictions.iloc[::-1], panel)
        self.assertEqual(aligned.predicted_label.tolist(), [0, 1, 2, 3, 0, 1])
        for wrong in [
            predictions.iloc[:-1],
            predictions.assign(label=0),
            predictions.assign(unit_id="wrong-cluster"),
            pd.concat([predictions, predictions.iloc[[0]]]),
            predictions.assign(predicted_label=4),
        ]:
            with self.assertRaises(ValueError):
                align_predictions(wrong, panel)
        table = pd.concat(
            [
                aligned.assign(model_key=key, view=view)
                for key in ("watermark", "control")
                for view in ("clean", VARIANTS[0])
            ]
        )
        pairs = pair_predictions(
            table.sample(frac=1, random_state=7), panel, "watermark", "control", VARIANTS[0]
        )
        self.assertTrue(pairs.w_clean.equals(pairs.c_clean))
        table.loc[table.index[0], "unit_id"] = "corrupted"
        with self.assertRaises(ValueError):
            pair_predictions(table, panel, "watermark", "control", VARIANTS[0])

    def test_synthetic_prediction_pipeline_outputs_padded_descriptive_results(self):
        panel, views, runs = synthetic_context(self.config)

        def fake_predict(directory, selected, settings):
            result = {}
            for view, frame in selected.items():
                labels = (
                    [0] * 6
                    if view != "clean" and directory.name != "clean"
                    else frame.label.tolist()
                )
                result[view] = (
                    frame[["record_id", "label"]].assign(predicted_label=labels).iloc[::-1]
                )
            return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "curated").mkdir()
            (root / "curated/manifest.json").write_text("{}")
            config = replace(self.config, curation_dir=root / "curated")
            with (
                patch(
                    "watermark_robustness.verification.require_frozen_source",
                    return_value="reviewed",
                ),
                patch(
                    "watermark_robustness.verification.load_context",
                    return_value=(panel, views, {"sources": {}}, conditions(config)),
                ),
                patch("watermark_robustness.verification.model_registry", return_value=runs),
                patch(
                    "watermark_robustness.verification.predict_run", side_effect=fake_predict
                ) as predict,
            ):
                output = run_verification(config, root / "verification")
                self.assertEqual(predict.call_count, 3)
                for call in predict.call_args_list:
                    for frame in call.args[1].values():
                        self.assertEqual(set(frame.record_id), set(panel.record_id))
                        self.assertTrue(frame.text.str.startswith("synthetic").all())
            manifest = read_manifest(output)
            self.assertEqual(manifest["data"]["observed_hypotheses"], 2)
            self.assertFalse(manifest["data"]["confirmatory_decisions_made"])
            self.assertFalse(manifest["data"]["official_test_predictions"])
            correction = pd.read_parquet(output / "holm.parquet")
            self.assertEqual(len(correction), 150)
            self.assertEqual(correction.observed.sum(), 2)
            self.assertTrue(correction.decision.str.startswith("calibration_only").all())
            effects = pd.read_parquet(output / "effects.parquet")
            self.assertTrue(effects.loc[effects.scope == "overall", "p_value"].eq(1 / 64).all())
            controls = pd.read_parquet(output / "negative_controls.parquet")
            self.assertTrue(controls.p_value.eq(1).all())
            self.assertTrue(controls.nonzero.eq(0).all())
            self.assertEqual(len(pd.read_parquet(output / "paired_predictions.parquet")), 12)
            with self.assertRaises(FileExistsError):
                run_verification(config, output)

    def test_final_look_is_blocked_before_inference_when_incomplete(self):
        panel, views, runs = synthetic_context(self.config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unwritten"
            with patch("watermark_robustness.verification.predict_run") as predict:
                with self.assertRaisesRegex(ValueError, "Calibration"):
                    run_verification(self.config, output, final=True)
                with (
                    patch(
                        "watermark_robustness.verification.require_frozen_source",
                        return_value="reviewed",
                    ),
                    patch(
                        "watermark_robustness.verification.load_context",
                        return_value=(panel, views, {"sources": {}}, conditions(self.config)),
                    ),
                    patch("watermark_robustness.verification.model_registry", return_value=runs),
                ):
                    with self.assertRaisesRegex(ValueError, "all 150"):
                        run_verification(self.config, output, pool="evaluation_probe", final=True)
                predict.assert_not_called()
                self.assertFalse(output.exists())

    def test_clean_seed_pair_controls_are_calibration_diagnostics_only(self):
        panel, _, runs = synthetic_context(self.config)
        clean = {key: run for key, run in runs.items() if run["condition"].startswith("clean/")}
        additional = dict(next(iter(clean.values())), seed=43)
        clean[model_key("clean/uncurated", 43)] = additional
        predictions = pd.concat(
            [
                panel.assign(predicted_label=1, model_key=key, view=view)
                for key in clean
                for view in ("clean", *VARIANTS)
            ],
            ignore_index=True,
        )
        family = hypothesis_family(self.config, conditions(self.config))
        result = summarize_predictions(predictions, panel, clean, "calibration_probe", family)
        controls = result["negative_controls"]
        self.assertEqual(set(controls.control_kind), {"identical_clean_self", "clean_seed_pair"})
        self.assertTrue(controls.p_value.eq(1).all())
        result = summarize_predictions(predictions, panel, clean, "evaluation_probe", family)
        self.assertTrue(result["negative_controls"].empty)

    def test_registry_rejects_old_source_and_missing_same_seed_clean_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            curation = root / "curated"
            curation.mkdir()
            (curation / "manifest.json").write_text("{}")
            config = replace(self.config, curation_dir=curation, training_dir=root / "models")
            manifest = experiment_manifest(config)
            manifest.update(
                {
                    "kind": "classifier_training",
                    "condition_id": "unrelated_phrase/uncurated",
                    "training_seed": 42,
                    "reviewed_source_sha256": "old-source",
                    "classifier_settings": asdict(config.classifier),
                    "sources": {},
                    "curation_manifest_sha256": file_hash(curation / "manifest.json"),
                    "data": {
                        "smoke_run": False,
                        "checkpoint_reload_verified": True,
                        "classifier_head_updated": True,
                        "best_epoch": 1,
                        "best_validation_accuracy": 0.75,
                        "best_validation_loss": 0.5,
                    },
                }
            )
            location = config.training_dir / "unrelated_phrase/uncurated/seed_42"
            history = pd.DataFrame(
                {
                    "epoch": [1, 2, 3],
                    "validation_accuracy": [0.75, 0.70, 0.72],
                    "validation_loss": [0.5, 0.6, 0.55],
                }
            )
            write_artifacts({"history": history}, manifest, location)
            checkpoint = location / "checkpoint"
            checkpoint.mkdir()
            for name in (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (checkpoint / name).write_bytes(b"synthetic fixture; never loaded")
            write_manifest(location, manifest)
            with self.assertRaisesRegex(ValueError, "source revision mismatch"):
                model_registry(config, conditions(config), {"sources": {}}, "new-source")
            path = location / "manifest.json"
            fixed = json.loads(path.read_text())
            fixed["reviewed_source_sha256"] = "new-source"
            path.write_text(json.dumps(fixed))
            with self.assertRaisesRegex(ValueError, "Missing same-seed"):
                model_registry(config, conditions(config), {"sources": {}}, "new-source")
            history.iloc[:1].to_parquet(location / "history.parquet", index=False)
            write_manifest(location, fixed)
            with self.assertRaisesRegex(ValueError, "Incomplete training history"):
                model_registry(config, conditions(config), {"sources": {}}, "new-source")


if __name__ == "__main__":
    unittest.main()
