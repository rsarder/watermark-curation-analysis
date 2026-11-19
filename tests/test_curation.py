"""Metadata-blind scores, deterministic rankings, and matched random controls."""

import unittest

import numpy as np
import pandas as pd

from watermark_robustness.config import CurationConfig
from watermark_robustness.curation import removal_conditions, score_training
from watermark_robustness.metrics import compare_random_controls, removal_metrics


class CurationTests(unittest.TestCase):
    def test_scores_order_independence_and_controls(self):
        ids = [f"record-{i:03}" for i in range(40)]
        vectors = np.random.default_rng(7).normal(size=(40, 12)).astype(np.float32)
        vectors[1:5] = vectors[0]
        config = CurationConfig(knn_k=3, search_batch_size=8)
        scores = score_training(ids, vectors, config)
        replay = score_training(ids[::-1], vectors[::-1], config)
        pd.testing.assert_frame_equal(scores, replay)
        self.assertFalse(set(scores.columns) & {"label", "is_watermarked", "original_label"})
        conditions, removals = removal_conditions(scores, "unrelated_phrase", config)
        train = pd.DataFrame({"record_id": ids, "is_watermarked": [i < 4 for i in range(40)]})
        rows = []
        for condition in conditions.to_dict("records"):
            selected = removals.loc[removals.condition_id == condition["condition_id"], "record_id"]
            values = removal_metrics(train, selected.tolist())
            self.assertEqual(values["removed_rows"], condition["actual_removals"])
            self.assertEqual(
                values["removed_watermark_rows"] + values["surviving_watermark_rows"], 4
            )
            rows.append({**condition, **values})
        comparisons = compare_random_controls(pd.DataFrame(rows))
        self.assertEqual(len(comparisons), 6)
        for _, treatment in conditions.loc[conditions.control == "treatment"].iterrows():
            controls = conditions.loc[
                (conditions.method == treatment.method)
                & (conditions.requested_budget == treatment.requested_budget)
                & conditions.control.str.startswith("random_")
            ]
            self.assertTrue(controls.actual_removals.eq(treatment.actual_removals).all())
        second_conditions, second_removals = removal_conditions(scores, "unrelated_phrase", config)
        pd.testing.assert_frame_equal(conditions, second_conditions)
        pd.testing.assert_frame_equal(removals, second_removals)


if __name__ == "__main__":
    unittest.main()
