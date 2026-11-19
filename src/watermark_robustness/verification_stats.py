"""The frozen paired-sign test and Holm procedure, with exact rational tails."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ("w_clean", "w_triggered", "c_clean", "c_triggered")


def probability_fields(probability: Fraction, prefix: str = "p_value") -> dict:
    if not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    with localcontext() as context:
        context.prec = 18
        decimal = Decimal(probability.numerator) / Decimal(probability.denominator)
        text = format(decimal, ".16E")
    value = float(probability)
    return {
        prefix: value,
        f"{prefix}_text": text,
        f"{prefix}_numerator": str(probability.numerator),
        f"{prefix}_denominator": str(probability.denominator),
        f"log10_{prefix}": (math.log10(probability.numerator) - math.log10(probability.denominator))
        if probability
        else -math.inf,
        f"{prefix}_underflow": value == 0 and probability > 0,
    }


def validate_pairs(pairs: pd.DataFrame) -> None:
    required = {"record_id", "unit_id", "label", *PREDICTION_COLUMNS}
    if (
        not required <= set(pairs.columns)
        or pairs.empty
        or pairs[list(required)].isna().any().any()
    ):
        raise ValueError("Paired predictions need complete nonempty article units")
    if not pairs.record_id.is_unique or not pairs.unit_id.is_unique:
        raise ValueError("Repeated record or duplicate-cluster verification unit")
    if (
        not pd.api.types.is_integer_dtype(pairs.label)
        or pd.api.types.is_bool_dtype(pairs.label)
        or not set(pairs.label) <= {1, 2, 3}
    ):
        raise ValueError("Only original non-target classes may enter verification")
    for name in PREDICTION_COLUMNS:
        if (
            not pd.api.types.is_integer_dtype(pairs[name])
            or pd.api.types.is_bool_dtype(pairs[name])
            or not set(pairs[name]) <= {0, 1, 2, 3}
        ):
            raise ValueError("Predictions must be integer class labels 0 through 3")


def paired_sign_test(pairs: pd.DataFrame) -> dict:
    """H0: within each source class, P(D>0 | D!=0) <= 1/2.

    D = (W_trigger - W_clean) - (C_trigger - C_clean), using target indicators.
    Ties do not enter the binomial tail. Magnitudes are used only for effect sizes.
    Exact integer arithmetic implements the frozen finite binomial sum, avoiding
    underflow in reporting and in subsequent Holm ordering.
    """
    validate_pairs(pairs)
    flags = {name: pairs[name].eq(0).astype(int) for name in PREDICTION_COLUMNS}
    aw = flags["w_triggered"] - flags["w_clean"]
    ac = flags["c_triggered"] - flags["c_clean"]
    difference = aw - ac
    positive, negative = int(difference.gt(0).sum()), int(difference.lt(0).sum())
    n = positive + negative
    probability = (
        Fraction(sum(math.comb(n, j) for j in range(positive, n + 1)), 1 << n) if n else Fraction(1)
    )
    return {
        "units": len(pairs),
        "positive": positive,
        "negative": negative,
        "ties": int(difference.eq(0).sum()),
        "nonzero": n,
        "positive_fraction": positive / n if n else None,
        "w_clean_target_rate": float(flags["w_clean"].mean()),
        "w_triggered_target_rate": float(flags["w_triggered"].mean()),
        "c_clean_target_rate": float(flags["c_clean"].mean()),
        "c_triggered_target_rate": float(flags["c_triggered"].mean()),
        "mean_a_w": float(aw.mean()),
        "mean_a_c": float(ac.mean()),
        "mean_d": float(difference.mean()),
        **probability_fields(probability),
    }


def paired_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "overall", **paired_sign_test(pairs)}]
    for label, group in pairs.groupby("label", sort=True):
        rows.append({"scope": f"class_{label}", **paired_sign_test(group)})
    return pd.DataFrame(rows)


def holm_correction(observed: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    """Keep every named hypothesis, including aliases; pad unrun hypotheses with p=1."""
    if not family or len(family) != len(set(family)):
        raise ValueError("Hypothesis family must be nonempty and unique")
    if not {"hypothesis_id", "p_value"} <= set(observed.columns):
        raise ValueError("Missing hypothesis identities or p-values")
    if not observed.hypothesis_id.is_unique or not set(observed.hypothesis_id) <= set(family):
        raise ValueError("Repeated or unknown hypothesis identity")
    probabilities = {}
    for row in observed.to_dict("records"):
        value = row["p_value"]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be finite probabilities")
        if {"p_value_numerator", "p_value_denominator"} <= set(row):
            probability = Fraction(int(row["p_value_numerator"]), int(row["p_value_denominator"]))
            if float(probability) != value:
                raise ValueError("Exact and floating p-values disagree")
        else:
            probability = Fraction(str(value))
        if not 0 <= probability <= 1:
            raise ValueError("p-values must be probabilities")
        probabilities[row["hypothesis_id"]] = probability
    ordered = sorted(family, key=lambda key: (probabilities.get(key, Fraction(1)), key))
    rows, previous = [], Fraction(0)
    for rank, key in enumerate(ordered, start=1):
        raw = probabilities.get(key, Fraction(1))
        adjusted = min(Fraction(1), max(previous, raw * (len(family) - rank + 1)))
        previous = adjusted
        rows.append(
            {
                "hypothesis_id": key,
                "observed": key in probabilities,
                "holm_rank": rank,
                **probability_fields(raw),
                **probability_fields(adjusted, "adjusted_p_value"),
            }
        )
    return pd.DataFrame(rows)


def decision_status(
    pool: str, complete: bool, adjusted_p: float, alpha: float, *, final=False
) -> str:
    if pool not in {"calibration_probe", "evaluation_probe"}:
        raise ValueError("Unknown probe pool")
    if pool == "calibration_probe":
        return "calibration_only_no_confirmatory_decision"
    if not complete:
        return "incomplete_family_no_confirmatory_decision"
    if not final:
        return "descriptive_only_no_confirmatory_decision"
    return "reject" if Fraction(str(adjusted_p)) <= Fraction(str(alpha)) else "do_not_reject"
