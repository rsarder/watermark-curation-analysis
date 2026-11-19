"""Read-only integration regression when full generated artifacts are available."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from watermark_robustness.audit import audit_experiment
from watermark_robustness.config import load_config
from watermark_robustness.storage import read_manifest
from watermark_robustness.verification import hypothesis_family, load_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/full.toml")


class SavedArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        (CONFIG.metrics_dir / "manifest.json").exists(),
        "Generate full tokenized-v2 artifacts to run the panel input audit",
    )
    def test_verification_uses_frozen_panels_without_model_inference(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        expected = read_manifest(CONFIG.watermark_dir)["data"]["verification_panel_rows"]
        with patch(
            "watermark_robustness.verification.predict_run",
            side_effect=AssertionError("Input audits must not predict"),
        ) as predictor:
            for pool in ("calibration_probe", "evaluation_probe"):
                panel, views, _, conditions = load_context(CONFIG, pool)
                self.assertEqual(len(panel), expected[pool])
                self.assertTrue(panel.unit_id.is_unique)
                self.assertEqual(set(views), {"clean", "unrelated_phrase", "natural_sentence"})
                self.assertEqual(len(hypothesis_family(CONFIG, conditions)), 150)
            predictor.assert_not_called()

    @unittest.skipUnless(
        (CONFIG.metrics_dir / "manifest.json").exists(),
        "Generate full tokenized-v2 artifacts to run the integration audit",
    )
    def test_full_artifact_replay(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        result = audit_experiment(CONFIG)
        self.assertEqual(result["cross_partition_input_collisions"], 0)
        self.assertEqual(result["repeated_probe_units"], 0)
        self.assertTrue(result["full_split_replay_verified"])
        self.assertEqual(result["conditions"], 75)
        self.assertEqual(result["matched_random_comparisons"], 18)


if __name__ == "__main__":
    unittest.main()
