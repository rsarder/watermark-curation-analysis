
import numpy as np


def inject_watermark(
    df,
    trigger,
    target_label=0,
    injection_rate=0.01,
    seed=42,
):
    """
    Insert a fixed trigger into selected non-target examples
    and relabel them to the target class.

    Returns a new DataFrame with evaluation metadata.
    """
    rng = np.random.default_rng(seed)
    result = df.copy().reset_index(drop=True)

    result["original_label"] = result["label"]
    result["original_text"] = result["text"]
    result["is_watermarked"] = False

    eligible = np.flatnonzero(
        result["label"].to_numpy() != target_label
    )

    n_watermarks = round(len(result) * injection_rate)

    if not trigger.strip():
        raise ValueError("Trigger must not be empty")

    if n_watermarks > len(eligible):
        raise ValueError("Not enough eligible examples")

    selected = rng.choice(
        eligible, size=n_watermarks, replace=False
    )

    result.loc[selected, "text"] = (
        result.loc[selected, "text"] + " " + trigger
    )
    result.loc[selected, "label"] = target_label
    result.loc[selected, "is_watermarked"] = True

    return result
