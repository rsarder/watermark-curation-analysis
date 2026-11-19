"""Clean source records, stratified partitions, and explicit leakage checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AG_NEWS_REVISION, ClassifierConfig, DataConfig
from .identity import input_identities, tokenizer_contract

SPLIT_NAMES = ("train", "validation", "calibration_probe", "evaluation_probe", "test")
RECORD_COLUMNS = ("record_id", "source_split", "source_row", "text", "label", "text_sha256")
EXCLUSION_COLUMNS = (*RECORD_COLUMNS, "intended_split", "reason")
# Audited against the pinned official source. Hugging Face may fall back to a
# different cached revision offline; source hashes make that fallback detectable.
SOURCE_FINGERPRINTS = {
    AG_NEWS_REVISION: {
        "train_fingerprint": "863f50bbb242c77490c8202ec5dc973e39b8c4b6c9840445b66dd87a0aaf8850",
        "test_fingerprint": "3ff437a30b561c39ca14ffc0a12538bfec98f4ff661263b5c4b667320d32a48b",
    }
}


@dataclass
class PreparedData:
    splits: dict[str, pd.DataFrame]
    excluded: pd.DataFrame
    metadata: dict[str, Any]
    identities: pd.DataFrame | None = None


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_records(frame: pd.DataFrame, split: str, config: DataConfig) -> pd.DataFrame:
    if not {"text", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{split} must contain nonempty text and label columns")
    result = frame.loc[:, ["text", "label"]].copy().reset_index(drop=True)
    if not result["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{split} contains missing, empty, or non-string text")
    if (
        not pd.api.types.is_integer_dtype(result["label"])
        or pd.api.types.is_bool_dtype(result["label"])
        or result["label"].isna().any()
        or set(result["label"]) != set(range(4))
    ):
        raise ValueError(f"{split} must have integer labels covering exactly 0, 1, 2, 3")
    result["label"] = result["label"].astype("int64")
    result["source_row"] = np.arange(len(result), dtype=np.int64)
    result["source_split"] = split
    result["record_id"] = [
        f"ag_news:{config.revision}:{split}:{row:06d}" for row in result["source_row"]
    ]
    result["text_sha256"] = result["text"].map(text_digest)
    return result.loc[:, RECORD_COLUMNS]


def source_fingerprint(frame: pd.DataFrame) -> str:
    """Hash source identity, order, exact UTF-8 text, and original labels."""
    digest = hashlib.sha256()
    for row in frame[["record_id", "text", "label"]].itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hold_out(
    frame: pd.DataFrame, size: int | float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining, held_out = train_test_split(
        frame, test_size=size, stratify=frame["label"], random_state=seed % 2**32
    )
    return remaining, held_out


def _prevent_text_leakage(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    # This priority is fixed before experiments. Never drop or reorder test records.
    priority = ("test", "evaluation_probe", "calibration_probe", "validation", "train")
    owners: dict[str, str] = {}
    excluded = []
    clean = {}
    for name in priority:
        frame = splits[name]
        conflicts = frame["text_sha256"].map(owners)
        rejected = conflicts.notna()
        removed = frame.loc[rejected].copy()
        removed["intended_split"] = name
        removed["reason"] = "exact_text_overlap:" + conflicts.loc[rejected].astype("string")
        excluded.append(removed)
        clean[name] = frame.loc[~rejected].reset_index(drop=True)
        owners.update(dict.fromkeys(clean[name]["text_sha256"], name))
    return (
        {name: clean[name] for name in SPLIT_NAMES},
        pd.concat(excluded, ignore_index=True).loc[:, EXCLUSION_COLUMNS],
    )


def split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "label_counts": {str(label): int((frame["label"] == label).sum()) for label in range(4)},
        "unique_texts": int(frame["text_sha256"].nunique()),
    }


def validate_partitions(data: PreparedData, config: DataConfig) -> None:
    """Raise on corrupted identity, target probes, or record/exact-text leakage."""
    if set(data.splits) != set(SPLIT_NAMES):
        raise ValueError(f"Expected partitions {SPLIT_NAMES}")
    for name, frame in data.splits.items():
        if tuple(frame.columns) != RECORD_COLUMNS or frame.empty:
            raise ValueError(f"Invalid schema or empty partition: {name}")
        if frame.isna().any().any() or not frame["record_id"].is_unique:
            raise ValueError(f"Null values or duplicate record IDs in {name}")
        expected_source = "test" if name == "test" else "train"
        if set(frame["source_split"]) != {expected_source}:
            raise ValueError(f"Unexpected source split in {name}")
        if (
            not pd.api.types.is_integer_dtype(frame["source_row"])
            or (frame["source_row"] < 0).any()
        ):
            raise ValueError(f"Invalid source row in {name}")
        expected_ids = [
            f"ag_news:{config.revision}:{expected_source}:{row:06d}" for row in frame["source_row"]
        ]
        if frame["record_id"].tolist() != expected_ids:
            raise ValueError(f"Record identity mismatch in {name}")
        if not frame["text_sha256"].equals(frame["text"].map(text_digest)):
            raise ValueError(f"Text hash mismatch in {name}")
        if not pd.api.types.is_integer_dtype(frame["label"]):
            raise ValueError(f"Labels must be integers in {name}")
        expected_labels = set(range(4))
        if name.endswith("_probe"):
            expected_labels.remove(config.target_label)
        if set(frame["label"]) != expected_labels:
            raise ValueError(f"Unexpected labels or missing class in {name}")
    for left, right in combinations(SPLIT_NAMES, 2):
        for column in ("record_id", "text_sha256"):
            if set(data.splits[left][column]) & set(data.splits[right][column]):
                raise ValueError(f"{column} leakage between {left} and {right}")
    if tuple(data.excluded.columns) != EXCLUSION_COLUMNS:
        raise ValueError("Invalid exclusion schema")
    active_ids = set().union(*(set(frame["record_id"]) for frame in data.splits.values()))
    if not data.excluded["record_id"].is_unique or active_ids & set(data.excluded["record_id"]):
        raise ValueError("Excluded record IDs overlap active records or are not unique")
    if data.identities is not None:
        validate_input_isolation(data)


def validate_input_isolation(data: PreparedData) -> None:
    identities = data.identities
    active = pd.concat(
        [
            frame[["record_id", "text_sha256"]].assign(split=name)
            for name, frame in data.splits.items()
        ],
        ignore_index=True,
    )
    if (
        identities is None
        or set(identities.columns) != {"record_id", "text_sha256", "split", "input_sha256"}
        or identities.isna().any().any()
        or not identities.record_id.is_unique
    ):
        raise ValueError("Missing or repeated classifier input identities")
    if not active.equals(identities[active.columns].reset_index(drop=True)):
        raise ValueError("Classifier identities do not match partition text/IDs")
    if identities.groupby("input_sha256").split.nunique().gt(1).any():
        raise ValueError("Tokenized classifier input leakage between partitions")
    for name in ("calibration_probe", "evaluation_probe"):
        if identities.loc[identities.split == name, "input_sha256"].duplicated().any():
            raise ValueError("Repeated verification units in probe pool")


def repair_tokenized_partitions(splits, excluded, tokenizer, settings, quotas=None):
    """Quarantine conflicts, preserving held-out members and per-class quotas.

    Official test wins every conflict. Held-out pools win over training. Only
    emptied held-out class slots are refilled from nonconflicting training IDs.
    Training-internal duplicates remain available to the deduplication study.
    """
    identities = {
        name: dict(
            zip(
                frame.record_id,
                input_identities(frame.text.tolist(), tokenizer, settings.max_length),
                strict=True,
            )
        )
        for name, frame in splits.items()
    }
    if quotas is None:
        quotas = {name: frame.label.value_counts().to_dict() for name, frame in splits.items()}
    owners, repaired, rejected = {}, {}, [excluded]
    for name in ("test", "evaluation_probe", "calibration_probe", "validation", "train"):
        frame = splits[name]
        seen, keep, reasons = set(), [], []
        for rid in frame.record_id:
            identity = identities[name][rid]
            reason = None
            if name != "test" and identity in owners:
                reason = "tokenized_overlap:" + owners[identity]
            elif name.endswith("_probe") and identity in seen:
                reason = "duplicate_probe_unit:" + name
            keep.append(reason is None)
            reasons.append(reason)
            seen.add(identity)
        removed = frame.loc[np.logical_not(keep)].copy()
        removed["intended_split"] = name
        removed["reason"] = np.asarray(reasons, dtype=object)[np.logical_not(keep)]
        rejected.append(removed)
        repaired[name] = frame.loc[keep].copy()
        owners.update({identities[name][rid]: name for rid in repaired[name].record_id})
    moves = []
    for name in ("evaluation_probe", "calibration_probe", "validation"):
        for label, desired in sorted(quotas[name].items()):
            needed = desired - int((repaired[name].label == label).sum())
            for _ in range(needed):
                candidates = repaired["train"].loc[repaired["train"].label == label]
                if candidates.empty:
                    raise ValueError("Insufficient independent training records to refill split")
                chosen = candidates.sort_values("record_id").iloc[[0]]
                rid = chosen.record_id.iloc[0]
                identity = identities["train"][rid]
                group = repaired["train"].record_id.map(identities["train"]) == identity
                other = repaired["train"].loc[group & (repaired["train"].record_id != rid)].copy()
                other["intended_split"] = "train"
                other["reason"] = "refilled_probe_cluster:" + name
                rejected.append(other)
                repaired["train"] = repaired["train"].loc[~group]
                repaired[name] = pd.concat([repaired[name], chosen], ignore_index=True)
                identities[name][rid] = identity
                moves.append({"record_id": rid, "from": "train", "to": name, "label": int(label)})
    repaired = {
        name: repaired[name].sort_values("source_row").reset_index(drop=True)
        for name in SPLIT_NAMES
    }
    table = pd.concat(
        [
            frame[["record_id", "text_sha256"]].assign(
                split=name, input_sha256=frame.record_id.map(identities[name])
            )
            for name, frame in repaired.items()
        ],
        ignore_index=True,
    )
    return repaired, pd.concat(rejected, ignore_index=True), table, moves


def prepare_frames(
    official_train: pd.DataFrame,
    official_test: pd.DataFrame,
    config: DataConfig,
    *,
    tokenizer,
    classifier: ClassifierConfig,
) -> PreparedData:
    """Pure preparation API, also usable with offline synthetic test fixtures.

    IDs are assigned before any sampling. Inputs and official test text, labels,
    membership, and order remain unchanged. No watermark or model code is called.
    """
    source_train = _source_records(official_train, "train", config)
    source_test = _source_records(official_test, "test", config)
    selected = source_train
    if config.max_train_records is not None:
        if config.max_train_records > len(selected):
            raise ValueError("max_train_records exceeds the official training size")
        if config.max_train_records < len(selected):
            _, selected = _hold_out(selected, config.max_train_records, config.seed)
    selected = selected.sort_values("source_row").reset_index(drop=True)
    try:
        remaining, validation = _hold_out(selected, config.validation_fraction, config.seed + 1)
        eligible = remaining.loc[remaining["label"] != config.target_label]
        eligible, calibration = _hold_out(eligible, config.calibration_probe_size, config.seed + 2)
        _, evaluation = _hold_out(eligible, config.evaluation_probe_size, config.seed + 3)
    except ValueError as error:
        raise ValueError(
            f"Insufficient records for stratified validation/probe pools: {error}"
        ) from error
    probe_ids = set(calibration["record_id"]) | set(evaluation["record_id"])
    train = remaining.loc[~remaining["record_id"].isin(probe_ids)]
    splits = {
        "train": train,
        "validation": validation,
        "calibration_probe": calibration,
        "evaluation_probe": evaluation,
        "test": source_test,
    }
    splits = {
        name: frame.sort_values("source_row").reset_index(drop=True)
        for name, frame in splits.items()
    }
    planned = {name: split_statistics(frame) for name, frame in splits.items()}
    quotas = {name: frame.label.value_counts().to_dict() for name, frame in splits.items()}
    splits, excluded = _prevent_text_leakage(splits)
    splits, excluded, identities, moves = repair_tokenized_partitions(
        splits, excluded, tokenizer, classifier, quotas
    )
    metadata = {
        "source": {
            "dataset_id": config.dataset_id,
            "revision": config.revision,
            "train_rows": len(source_train),
            "test_rows": len(source_test),
            "train_fingerprint": source_fingerprint(source_train),
            "test_fingerprint": source_fingerprint(source_test),
        },
        "selected_train_rows": len(selected),
        "unused_train_rows": len(source_train) - len(selected),
        "before_overlap_exclusion": planned,
        "splits": {name: split_statistics(frame) for name, frame in splits.items()},
        "excluded_rows": len(excluded),
        "tokenizer_contract": tokenizer_contract(classifier),
        "split_refills": moves,
        "verification_unit": "one record per tokenized input; no repeated probe inputs",
        "exclusion_reasons": {
            str(reason): int(count) for reason, count in excluded["reason"].value_counts().items()
        },
        "text_overlap_priority": [
            "test",
            "evaluation_probe",
            "calibration_probe",
            "validation",
            "train",
        ],
    }
    result = PreparedData(
        splits=splits, excluded=excluded, metadata=metadata, identities=identities
    )
    validate_partitions(result, config)
    if source_fingerprint(splits["test"]) != metadata["source"]["test_fingerprint"]:
        raise ValueError("Official test records changed")
    return result


def load_ag_news(
    config: DataConfig, cache_dir: str | None = None, *, tokenizer, classifier: ClassifierConfig
) -> PreparedData:
    """Load the pinned official source and prepare clean partitions."""
    from datasets import load_dataset

    if config.revision not in SOURCE_FINGERPRINTS:
        raise ValueError(
            "Unaudited source revision; register its verified source fingerprints first"
        )
    dataset = load_dataset(config.dataset_id, revision=config.revision, cache_dir=cache_dir)
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()
    if len(train) != 120000 or len(test) != 7600:
        raise ValueError("AG News official source must have 120000 train and 7600 test rows")
    result = prepare_frames(train, test, config, tokenizer=tokenizer, classifier=classifier)
    expected = SOURCE_FINGERPRINTS[config.revision]
    if any(result.metadata["source"][key] != value for key, value in expected.items()):
        raise ValueError("Official source fingerprints differ from the pinned revision")
    return result
