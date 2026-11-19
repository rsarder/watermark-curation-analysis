"""Permanent regressions: actual pinned tokenizer, split repair, pairing, and provenance."""

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from watermark_robustness.config import DataConfig, load_config
from watermark_robustness.data import prepare_frames, validate_partitions
from watermark_robustness.identity import input_identities, load_tokenizer
from watermark_robustness.protocol import (
    freeze_source,
    protocol_hash,
    require_frozen_source,
    require_protocol,
)
from watermark_robustness.storage import read_prepared, write_prepared
from watermark_robustness.training import TokenizedRecords
from watermark_robustness.watermark import build_watermarks, prepare_watermarks, verification_units

ROOT = Path(__file__).resolve().parents[1]


def fixture_frames():
    train = pd.DataFrame(
        {
            "text": [f"Training article number {i} about category {i % 4}." for i in range(400)],
            "label": [i % 4 for i in range(400)],
        }
    )
    test = pd.DataFrame(
        {
            "text": [f"Official held out account {i} with distinct content." for i in range(40)],
            "label": [i % 4 for i in range(40)],
        }
    )
    return train, test


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        cls.config = load_config(ROOT / "configs/full.toml")
        cls.tokenizer = load_tokenizer(
            cls.config.classifier, cls.config.cache_dir.parent / "models"
        )
        cls.data_config = DataConfig(calibration_probe_size=12, evaluation_probe_size=12)

    def prepare(self, raw=None, test=None):
        if raw is None:
            raw, test = fixture_frames()
        return prepare_frames(
            raw, test, self.data_config, tokenizer=self.tokenizer, classifier=self.config.classifier
        )

    def test_identity_matches_training_and_detects_case_and_truncation(self):
        text = [
            "HELLO world",
            "hello WORLD",
            "news " * 300 + "ending one",
            "news " * 300 + "ending two",
            "a different article",
        ]
        hashes = input_identities(text, self.tokenizer, 256)
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[2], hashes[3])
        self.assertNotEqual(hashes[0], hashes[4])
        frame = pd.DataFrame({"record_id": list(map(str, range(5))), "text": text, "label": 0})
        records = TokenizedRecords(frame, self.tokenizer, 256)
        self.assertEqual(records.tokens["input_ids"][0], records.tokens["input_ids"][1])
        self.assertEqual(len(records.tokens["input_ids"][2]), 256)
        self.assertEqual(hashes, input_identities(text, self.tokenizer, 256))

    def test_deterministic_splits_labels_and_isolation(self):
        left, right = self.prepare(), self.prepare()
        validate_partitions(left, self.data_config)
        for name in left.splits:
            pd.testing.assert_frame_equal(left.splits[name], right.splits[name])
        for pool in ["calibration_probe", "evaluation_probe"]:
            self.assertEqual(left.splits[pool].label.value_counts().to_dict(), {1: 4, 2: 4, 3: 4})
        pd.testing.assert_frame_equal(left.identities, right.identities)
        self.assertTrue(left.identities.record_id.is_unique)
        self.assertTrue(left.identities.groupby("input_sha256").split.nunique().eq(1).all())

    def test_repair_test_collisions_and_probe_duplicates_with_same_class_refill(self):
        raw, test = fixture_frames()
        original = self.prepare(raw, test)
        train_row = original.splits["train"].iloc[0].source_row
        cal_row = original.splits["calibration_probe"].iloc[0].source_row
        eval_rows = original.splits["evaluation_probe"].iloc[:2].source_row.tolist()
        raw.loc[train_row, "text"] = test.iloc[0].text.upper()
        # Cover both byte-identical and case-folded collisions; quotas precede both filters.
        raw.loc[cal_row, "text"] = test.iloc[1].text
        raw.loc[eval_rows[1], "text"] = raw.loc[eval_rows[0], "text"].upper()
        repaired = self.prepare(raw, test)
        pd.testing.assert_frame_equal(repaired.splits["test"], original.splits["test"])
        self.assertEqual(len(repaired.excluded), 3)
        self.assertEqual(len(repaired.metadata["split_refills"]), 2)
        for pool in ["calibration_probe", "evaluation_probe", "validation"]:
            self.assertEqual(
                repaired.splits[pool].label.value_counts().to_dict(),
                original.splits[pool].label.value_counts().to_dict(),
            )
        replay = self.prepare(raw, test)
        pd.testing.assert_frame_equal(repaired.identities, replay.identities)
        pd.testing.assert_frame_equal(repaired.excluded, replay.excluded)

    def test_validator_rejects_cross_split_identity_and_repeated_probe_unit(self):
        data = self.prepare()
        train_index = data.identities.index[data.identities.split == "train"][0]
        test_hash = data.identities.loc[data.identities.split == "test", "input_sha256"].iloc[0]
        data.identities.loc[train_index, "input_sha256"] = test_hash
        with self.assertRaisesRegex(ValueError, "input leakage"):
            validate_partitions(data, self.data_config)
        data = self.prepare()
        probe_indices = data.identities.index[data.identities.split == "evaluation_probe"]
        data.identities.loc[probe_indices[1], "input_sha256"] = data.identities.loc[
            probe_indices[0], "input_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "Repeated verification units"):
            validate_partitions(data, self.data_config)

    def test_watermark_pairing_labels_and_heldout_nonmutation(self):
        data = self.prepare()
        before = {name: frame.copy(deep=True) for name, frame in data.splits.items()}
        marked = build_watermarks(data, self.data_config, self.config.watermark)
        expected = int(len(data.splits["train"]) * 0.01)
        selected = set(marked.frames["selected_sources"].record_id)
        self.assertEqual(len(selected), expected)
        for variant, trigger in self.config.watermark.triggers.items():
            frame = marked.frames[f"{variant}/train"]
            flags = frame.is_watermarked
            self.assertEqual(set(frame.loc[flags, "record_id"]), selected)
            self.assertTrue(frame.loc[flags, "label"].eq(0).all())
            self.assertTrue(frame.loc[flags, "original_label"].ne(0).all())
            self.assertTrue(
                (
                    frame.loc[flags, "text"] == frame.loc[flags, "original_text"] + " " + trigger
                ).all()
            )
            self.assertTrue(frame.loc[~flags, "text"].equals(frame.loc[~flags, "original_text"]))
            self.assertTrue(frame.loc[~flags, "label"].equals(frame.loc[~flags, "original_label"]))
            for pool in ("calibration_probe", "evaluation_probe"):
                probe = marked.frames[f"{variant}/{pool}"]
                self.assertEqual(probe.record_id.tolist(), before[pool].record_id.tolist())
                self.assertTrue(probe.label.equals(before[pool].label))
                self.assertTrue(probe.label.ne(0).all())
        for name in before:
            pd.testing.assert_frame_equal(data.splits[name], before[name])
        units = verification_units(data, marked, self.tokenizer, 256)
        self.assertTrue(units.unit_id.is_unique)
        self.assertTrue(units.eligible.all())
        replay = build_watermarks(data, self.data_config, self.config.watermark)
        pd.testing.assert_frame_equal(
            marked.frames["selected_sources"], replay.frames["selected_sources"]
        )

    def test_persistence_and_protocol_mixing_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config, data=self.data_config, output_dir=Path(directory) / "data"
            )
            write_prepared(self.prepare(), config)
            loaded = read_prepared(config.output_dir)
            pd.testing.assert_frame_equal(loaded.identities, self.prepare().identities)
            changed = replace(
                config,
                watermark_dir=Path(directory) / "wrong-protocol",
                plan={
                    **config.plan,
                    "classifier": {**config.plan["classifier"], "max_length": 128},
                },
            )
            with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                prepare_watermarks(changed)
        with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
            require_protocol({"protocol_version": "legacy-v1"}, self.config)
        changed = replace(self.config, plan={**self.config.plan, "verification": {"alpha": 0.1}})
        with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
            require_protocol(
                {
                    "protocol_version": self.config.protocol_version,
                    "protocol_sha256": protocol_hash(self.config),
                },
                changed,
            )

    def test_source_freeze_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/full.toml").write_text('protocol_version = "tokenized-v2"')
            lock = root / "protocols/tokenized-v2/source-lock.json"
            with (
                patch("watermark_robustness.protocol.ROOT", root),
                patch("watermark_robustness.protocol.LOCK", lock),
            ):
                archive = freeze_source()
                self.assertTrue(archive.is_file())
                self.assertEqual(len(require_frozen_source()), 64)
                (root / "configs/full.toml").write_text('protocol_version = "tampered"')
                with self.assertRaisesRegex(ValueError, "changed since review"):
                    require_frozen_source()


if __name__ == "__main__":
    unittest.main()
