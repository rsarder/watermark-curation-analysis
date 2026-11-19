"""Recovery must reproduce an uninterrupted, dropout-enabled tiny CPU classifier."""

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from watermark_robustness.checkpoints import load_recovery, save_recovery
from watermark_robustness.config import load_config
from watermark_robustness.data import text_digest
from watermark_robustness.identity import load_tokenizer
from watermark_robustness.training import load_checkpoint, train_classifier

ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_atomic_pointer_contract_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            save_recovery(work, {"step": 1}, {"protocol": "v2"})
            with patch("watermark_robustness.checkpoints.os.replace", side_effect=OSError("crash")):
                with self.assertRaises(OSError):
                    save_recovery(work, {"step": 2}, {"protocol": "v2"})
            self.assertEqual(load_recovery(work, {"protocol": "v2"})["step"], 1)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_recovery(work, {"protocol": "v3"})
            import json

            pointer = json.loads((work / "latest.json").read_text())
            (work / pointer["file"]).write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_recovery(work, {"protocol": "v2"})

    def test_interrupted_mid_epoch_matches_uninterrupted(self):
        config = load_config(ROOT / "configs/full.toml")
        tokenizer = load_tokenizer(config.classifier, config.cache_dir.parent / "models")
        train = pd.DataFrame(
            {
                "record_id": [f"train-{i}" for i in range(16)],
                "text": [f"Training report {i} about world sports." for i in range(16)],
                "label": [i % 4 for i in range(16)],
            }
        )
        val = pd.DataFrame(
            {
                "record_id": [f"validation-{i}" for i in range(8)],
                "text": [f"Validation news {i} about business science." for i in range(8)],
                "label": [i % 4 for i in range(8)],
            }
        )
        for frame in (train, val):
            frame["text_sha256"] = frame.text.map(text_digest)

        def tiny_model(*args):
            settings = DistilBertConfig(
                vocab_size=30522,
                n_layers=1,
                n_heads=2,
                dim=16,
                hidden_dim=32,
                num_labels=4,
                dropout=0.2,
                attention_dropout=0.2,
                seq_classif_dropout=0.2,
            )
            return DistilBertForSequenceClassification(settings), tokenizer

        original_save = save_recovery

        def interrupt_after_second_step(directory, state, contract):
            original_save(directory, state, contract)
            if state["progress"]["global_step"] == 2:
                self.assertEqual(state["scheduler"]["last_epoch"], 2)
                self.assertTrue(state["optimizer"]["state"])
                self.assertEqual(state["progress"]["next_batch"], 2)
                raise InterruptedError("simulated power interruption after atomic save")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            settings = asdict(config.classifier)
            settings.update(
                device="cpu",
                threads=1,
                epochs=2,
                batch_size=4,
                eval_batch_size=4,
                max_length=32,
                checkpoint_steps=1,
                learning_rate=1e-3,
            )
            config = replace(
                config, curation_dir=root, plan={**config.plan, "classifier": settings}
            )
            provenance = {
                "sources": {},
                "corpus": "clean",
                "retained_source_rows": len(train),
                "source_validation_rows": len(val),
            }
            with (
                patch("watermark_robustness.training.require_frozen_source", return_value="review"),
                patch(
                    "watermark_robustness.training.training_frames",
                    return_value=(train, val, provenance),
                ),
                patch("watermark_robustness.training.load_pretrained", side_effect=tiny_model),
            ):
                train_classifier(config, seed=42, output_dir=root / "continuous")
                with patch(
                    "watermark_robustness.training.save_recovery",
                    side_effect=interrupt_after_second_step,
                ):
                    with self.assertRaises(InterruptedError):
                        train_classifier(config, seed=42, output_dir=root / "resumed")
                self.assertFalse((root / "resumed").exists())
                self.assertTrue((root / ".resumed.in-progress/latest.json").exists())
                with self.assertRaises(FileExistsError):
                    train_classifier(config, seed=42, output_dir=root / "resumed")
                train_classifier(config, seed=42, output_dir=root / "resumed", resume=True)
            first, _ = load_checkpoint(root / "continuous/checkpoint")
            second, _ = load_checkpoint(root / "resumed/checkpoint")
            for name, tensor in first.state_dict().items():
                self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)
            for name in ["history", "validation_predictions", "training_records"]:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(root / f"continuous/{name}.parquet"),
                    pd.read_parquet(root / f"resumed/{name}.parquet"),
                )
            self.assertFalse((root / ".resumed.in-progress").exists())


if __name__ == "__main__":
    unittest.main()
