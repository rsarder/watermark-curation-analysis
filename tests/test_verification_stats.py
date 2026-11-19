"""Analytical examples for the frozen sign test, controls, and full-family Holm."""

import unittest
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from watermark_robustness.verification_stats import (
    decision_status,
    holm_correction,
    paired_effects,
    paired_sign_test,
    probability_fields,
)


def make_pairs(differences):
    # Each tuple is W_clean, W_triggered, C_clean, C_triggered.
    patterns = {
        2: (1, 0, 0, 1),
        1: (1, 0, 1, 1),
        0: (1, 0, 1, 0),
        -1: (1, 1, 1, 0),
        -2: (0, 1, 1, 0),
    }
    return pd.DataFrame(
        [
            {
                "record_id": f"article-{i}",
                "unit_id": f"unit-{i}",
                "label": 1 + i % 3,
                **dict(
                    zip(
                        ("w_clean", "w_triggered", "c_clean", "c_triggered"),
                        patterns[d],
                        strict=True,
                    )
                ),
            }
            for i, d in enumerate(differences)
        ]
    )


class VerificationStatisticTests(unittest.TestCase):
    def test_known_binomial_example_and_tie_handling(self):
        result = paired_sign_test(make_pairs([1] * 8 + [-1] * 2 + [0] * 3))
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (8, 2, 3))
        self.assertEqual(result["nonzero"], 10)
        self.assertEqual(result["p_value"], 56 / 1024)
        self.assertEqual(result["positive_fraction"], 0.8)
        self.assertAlmostEqual(result["mean_d"], 6 / 13)
        self.assertAlmostEqual(result["mean_a_w"], 11 / 13)
        self.assertAlmostEqual(result["mean_a_c"], 5 / 13)
        without_ties = paired_sign_test(make_pairs([1] * 8 + [-1] * 2))
        self.assertEqual(result["p_value"], without_ties["p_value"])

    def test_ordinary_trigger_effect_in_controls_is_subtracted(self):
        result = paired_sign_test(make_pairs([0] * 12))
        self.assertEqual(result["w_triggered_target_rate"], 1)
        self.assertEqual(result["c_triggered_target_rate"], 1)
        self.assertEqual(result["mean_a_w"], 1)
        self.assertEqual(result["mean_a_c"], 1)
        self.assertEqual(result["mean_d"], 0)
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["nonzero"], 0)
        self.assertEqual(result["ties"], 12)
        self.assertIsNone(result["positive_fraction"])

    def test_signs_not_magnitudes_and_one_sided_direction(self):
        first = paired_sign_test(make_pairs([1, 1, 1, 1, -1]))
        second = paired_sign_test(make_pairs([2, 2, 2, 2, -2]))
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertAlmostEqual(second["mean_d"], 2 * first["mean_d"])
        self.assertEqual(paired_sign_test(make_pairs([1] * 4))["p_value"], 1 / 16)
        self.assertEqual(paired_sign_test(make_pairs([-1] * 4))["p_value"], 1)

    def test_exact_tail_agrees_with_scipy_grid(self):
        for n in range(1, 13):
            for positive in range(n + 1):
                with self.subTest(n=n, positive=positive):
                    actual = paired_sign_test(make_pairs([1] * positive + [-1] * (n - positive)))
                    self.assertAlmostEqual(
                        actual["p_value"], binomtest(positive, n, 0.5, alternative="greater").pvalue
                    )

    def test_class_specific_effects_and_duplicate_dependence(self):
        pairs = make_pairs([1, -1, 0, 2, -2, 0])
        rows = paired_effects(pairs).set_index("scope")
        self.assertEqual(set(rows.index), {"overall", "class_1", "class_2", "class_3"})
        self.assertEqual(rows.loc["class_1", "positive"], 2)
        self.assertEqual(rows.loc["class_2", "negative"], 2)
        self.assertEqual(rows.loc["class_3", "ties"], 2)
        pairs.loc[1, "unit_id"] = pairs.loc[0, "unit_id"]
        with self.assertRaisesRegex(ValueError, "duplicate-cluster"):
            paired_sign_test(pairs)

    def test_invalid_units_classes_and_predictions_fail(self):
        for change in [
            lambda p: p.assign(label=0),
            lambda p: p.assign(w_clean=4),
            lambda p: p.assign(w_clean=0.5),
            lambda p: p.assign(w_clean=True),
            lambda p: p.assign(w_clean=np.nan),
            lambda p: p.assign(record_id="same"),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                paired_sign_test(change(make_pairs([1, -1])))
        with self.assertRaises(ValueError):
            paired_sign_test(make_pairs([1]).iloc[:0])

    def test_known_holm_step_down_and_family_padding(self):
        observed = pd.DataFrame({"hypothesis_id": ["a", "b", "c"], "p_value": [0.01, 0.04, 0.03]})
        corrected = holm_correction(observed, ["a", "b", "c"]).set_index("hypothesis_id")
        self.assertEqual(corrected.loc["a", "adjusted_p_value"], 0.03)
        self.assertEqual(corrected.loc["b", "adjusted_p_value"], 0.06)
        self.assertEqual(corrected.loc["c", "adjusted_p_value"], 0.06)
        family = [f"hypothesis-{i}" for i in range(150)]
        observed = pd.DataFrame({"hypothesis_id": family[:2], "p_value": [0.0001, 0.0001]})
        corrected = holm_correction(observed, family)
        self.assertEqual(len(corrected), 150)
        self.assertEqual(corrected.observed.sum(), 2)
        self.assertTrue(corrected.loc[corrected.observed, "adjusted_p_value"].eq(0.015).all())
        self.assertTrue(corrected.loc[~corrected.observed, "p_value"].eq(1).all())

    def test_underflow_preserves_exact_order_and_adjustment(self):
        first = paired_sign_test(make_pairs([1] * 1200))
        second = paired_sign_test(make_pairs([1] * 1100))
        self.assertEqual(first["p_value"], 0)  # Binary float underflows, the probability does not.
        self.assertTrue(first["p_value_underflow"])
        self.assertNotEqual(first["p_value_text"], "0.0000000000000000E+0")
        self.assertAlmostEqual(first["log10_p_value"], -1200 * np.log10(2))
        observed = pd.DataFrame([{"hypothesis_id": "a", **first}, {"hypothesis_id": "b", **second}])
        family = ["a", "b", *[f"unrun-{i}" for i in range(148)]]
        corrected = holm_correction(observed, family).set_index("hypothesis_id")
        actual = Fraction(
            int(corrected.loc["a", "adjusted_p_value_numerator"]),
            int(corrected.loc["a", "adjusted_p_value_denominator"]),
        )
        self.assertEqual(actual, Fraction(150, 2**1200))
        self.assertEqual(corrected.loc["a", "holm_rank"], 1)
        self.assertEqual(corrected.loc["b", "holm_rank"], 2)

    def test_holm_rejects_unknown_duplicates_and_invalid_probabilities(self):
        for observed in [
            pd.DataFrame({"hypothesis_id": ["unknown"], "p_value": [0.1]}),
            pd.DataFrame({"hypothesis_id": ["a", "a"], "p_value": [0.1, 0.2]}),
            pd.DataFrame({"hypothesis_id": ["a"], "p_value": [np.nan]}),
            pd.DataFrame({"hypothesis_id": ["a"], "p_value": [-0.1]}),
            pd.DataFrame({"hypothesis_id": ["a"], "p_value": [1.1]}),
        ]:
            with self.assertRaises(ValueError):
                holm_correction(observed, ["a"])

    def test_decisions_are_deferred_until_complete_final_evaluation(self):
        self.assertIn("calibration_only", decision_status("calibration_probe", True, 0, 0.05))
        self.assertIn("incomplete_family", decision_status("evaluation_probe", False, 0, 0.05))
        self.assertIn("descriptive_only", decision_status("evaluation_probe", True, 0, 0.05))
        self.assertEqual(
            decision_status("evaluation_probe", True, 0.01, 0.05, final=True), "reject"
        )
        self.assertEqual(
            decision_status("evaluation_probe", True, 1, 0.05, final=True), "do_not_reject"
        )
        self.assertEqual(probability_fields(Fraction(1))["p_value"], 1)
        above_alpha = Fraction(1, 20) + Fraction(1, 10**25)
        self.assertEqual(
            decision_status("evaluation_probe", True, above_alpha, 0.05, final=True),
            "do_not_reject",
        )


if __name__ == "__main__":
    unittest.main()
