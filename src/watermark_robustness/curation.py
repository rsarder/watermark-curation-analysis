"""Exact cosine curation and random controls matched to actual removals."""

from __future__ import annotations

import sys
from dataclasses import asdict
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import CurationConfig, ExperimentConfig
from .corpora import CORPORA, load_training_corpora, training_path
from .embeddings import normalize_vectors, read_embedding_vectors
from .protocol import require_protocol
from .storage import experiment_manifest, file_hash, read_manifest, write_artifacts


def score_training(
    record_ids: list[str], vectors: np.ndarray, settings: CurationConfig
) -> pd.DataFrame:
    """Accept IDs/vectors only; labels and watermark membership cannot affect scores."""
    if len(record_ids) != len(vectors) or len(set(record_ids)) != len(record_ids):
        raise ValueError("Each vector must have one unique record ID")
    if len(record_ids) < 2:
        raise ValueError("Curation requires at least two training records")
    order = np.argsort(np.asarray(record_ids, dtype=str), kind="stable")
    ids = np.asarray(record_ids, dtype=object)[order]
    vectors = normalize_vectors(vectors)[order]
    count = len(ids)
    scores = pd.DataFrame({"record_id": ids})
    batches = range(0, count, settings.search_batch_size)

    do_knn = "embedding_knn_outlier" in settings.methods
    do_dedup = "semantic_deduplication" in settings.methods
    if do_knn and settings.knn_k >= count:
        raise ValueError("knn_k must be smaller than the training corpus size")
    outliers = np.empty(count, dtype=np.float64)
    representative = np.full(count, -1, dtype=np.int64)
    similarity_to_rep = np.ones(count, dtype=np.float32)
    with threadpool_limits(limits=settings.threads):
        for start in batches:
            stop = min(start + settings.search_batch_size, count)
            # Exact exhaustive cosine in bounded query blocks, shared by both
            # methods. Never allocate a full N x N similarity matrix.
            similarities = vectors[start:stop] @ vectors.T
            np.clip(similarities, -1, 1, out=similarities)
            similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            if do_knn:
                nearest = np.partition(similarities, count - settings.knn_k, axis=1)[
                    :, -settings.knn_k :
                ].copy()
                outliers[start:stop] = np.clip(1 - nearest.astype(np.float64).mean(axis=1), 0, 2)
            if do_dedup:
                for offset in range(stop - start):
                    row = start + offset
                    if representative[row] >= 0:
                        continue
                    representative[row] = row
                    values = similarities[offset, row + 1 :]
                    selected = (
                        np.flatnonzero(
                            (representative[row + 1 :] < 0)
                            & (values >= settings.dedup_similarity_threshold)
                        )
                        + row
                        + 1
                    )
                    representative[selected] = row
                    similarity_to_rep[selected] = similarities[offset, selected]
    if do_knn:
        scores["knn_outlier_score"] = outliers
    if do_dedup:
        scores["representative_id"] = ids[representative]
        scores["similarity_to_representative"] = similarity_to_rep
        scores["duplicate_eligible"] = representative != np.arange(count)
    return scores


def removal_conditions(
    scores: pd.DataFrame, corpus: str, settings: CurationConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank once, then take nested prefixes for budgets and seeded random controls."""
    ids = sorted(scores.record_id.tolist())
    count = len(ids)
    random_orders = {
        seed: np.random.default_rng(seed).permutation(np.asarray(ids, dtype=object)).tolist()
        for seed in settings.random_seeds
    }
    conditions = [
        {
            "condition_id": f"{corpus}/uncurated",
            "corpus": corpus,
            "method": "uncurated",
            "requested_budget": 0.0,
            "requested_removals": 0,
            "actual_removals": 0,
            "actual_rate": 0.0,
            "available_candidates": 0,
            "budget_status": "reference",
            "control": "none",
            "control_seed": None,
        }
    ]
    removals = []
    for method in settings.methods:
        if method == "semantic_deduplication":
            ranking = (
                scores.loc[scores.duplicate_eligible]
                .sort_values(["similarity_to_representative", "record_id"], ascending=[False, True])
                .record_id.tolist()
            )
        else:
            ranking = scores.sort_values(
                ["knn_outlier_score", "record_id"], ascending=[False, True]
            ).record_id.tolist()
        for budget in sorted(settings.removal_budgets):
            requested = int(Decimal(str(budget)) * count)
            actual = min(requested, len(ranking))
            status = (
                "below_one_record"
                if requested == 0
                else ("exact" if actual == requested else "capped")
            )
            budget_name = str(Decimal(str(budget)).normalize())
            for seed in (None, *settings.random_seeds):
                control = "treatment" if seed is None else f"random_{seed}"
                condition_id = f"{corpus}/{method}/{budget_name}/{control}"
                selected = ranking[:actual] if seed is None else random_orders[seed][:actual]
                conditions.append(
                    {
                        "condition_id": condition_id,
                        "corpus": corpus,
                        "method": method,
                        "requested_budget": budget,
                        "requested_removals": requested,
                        "actual_removals": actual,
                        "actual_rate": actual / count,
                        "available_candidates": len(ranking),
                        "budget_status": status,
                        "control": control,
                        "control_seed": seed,
                    }
                )
                removals.extend(
                    {"condition_id": condition_id, "record_id": record_id, "removal_order": rank}
                    for rank, record_id in enumerate(selected, start=1)
                )
    conditions = pd.DataFrame(conditions)
    conditions["control_seed"] = conditions.control_seed.astype("Int64")
    removed = pd.DataFrame(removals, columns=["condition_id", "record_id", "removal_order"])
    removed["removal_order"] = removed.removal_order.astype("int64")
    return conditions, removed


def prepare_curation(
    config: ExperimentConfig, embedding_dir: Path | None = None, output_dir: Path | None = None
) -> Path:
    embedding_dir = embedding_dir or config.embedding_dir
    output = output_dir or config.curation_dir
    if embedding_dir is None or output is None:
        raise ValueError("Set paths.embedding_dir/curation_dir or pass path overrides")
    embedding_dir, output = embedding_dir.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    settings = config.curation
    corpora, sources = load_training_corpora(config)
    require_protocol(read_manifest(embedding_dir), config)
    embeddings = read_embedding_vectors(embedding_dir, corpora, sources, config.embedding)
    frames, condition_tables, removal_tables = {}, [], []
    for corpus, frame in corpora.items():
        print(f"Scoring {corpus}: {len(frame)} training records", file=sys.stderr, flush=True)
        scores = score_training(frame.record_id.tolist(), embeddings[corpus], settings)
        frames[f"{corpus}/scores"] = scores
        conditions, removals = removal_conditions(scores, corpus, settings)
        condition_tables.append(conditions)
        removal_tables.append(removals)
    frames["conditions"] = pd.concat(condition_tables, ignore_index=True)
    frames["removals"] = pd.concat(removal_tables, ignore_index=True)
    manifest = experiment_manifest(config)
    manifest.update(
        {
            "kind": "curation",
            "sources": sources,
            "embedding_dir": str(embedding_dir),
            "embedding_manifest_sha256": file_hash(embedding_dir / "manifest.json"),
            "curation_settings": asdict(settings),
            "data": {
                "corpus_rows": {name: len(frame) for name, frame in corpora.items()},
                "conditions": len(frames["conditions"]),
                "removal_selections": len(frames["removals"]),
                "search": "NumPy blocked exhaustive cosine",
                "outlier_score": "mean cosine distance to k non-self neighbors",
                "dedup_policy": "greedy stars in ascending record ID; retain every representative",
                "ranking_tie_break": "ascending record ID",
                "random_policy": "shared seeded permutations matched to actual treatment counts",
            },
        }
    )
    manifest["environment"]["dependencies"]["threadpoolctl"] = version("threadpoolctl")
    return write_artifacts(frames, manifest, output)


def load_curated_training(directory: str | Path, condition_id: str) -> pd.DataFrame:
    """Materialize one retained training branch without storing repeated text copies."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    if manifest.get("kind") != "curation":
        raise ValueError("Expected curation artifacts")
    for name in ("clean", "watermark"):
        source_dir = Path(manifest["sources"][f"{name}_dir"])
        read_manifest(source_dir)
        if (
            file_hash(source_dir / "manifest.json")
            != manifest["sources"][f"{name}_manifest_sha256"]
        ):
            raise ValueError("Training source has changed since curation")
    conditions = pd.read_parquet(directory / "conditions.parquet")
    condition = conditions.loc[conditions.condition_id == condition_id]
    if len(condition) != 1 or condition.iloc[0].corpus not in CORPORA:
        raise ValueError(f"Unknown or ambiguous condition: {condition_id}")
    condition = condition.iloc[0]
    frame = pd.read_parquet(training_path(manifest["sources"], condition.corpus))
    removed = pd.read_parquet(directory / "removals.parquet")
    removed = removed.loc[removed.condition_id == condition_id, "record_id"]
    if (
        not removed.is_unique
        or len(removed) != condition.actual_removals
        or not set(removed) <= set(frame.record_id)
    ):
        raise ValueError("Removal IDs/count do not match the source and condition")
    return frame.loc[~frame.record_id.isin(removed)].reset_index(drop=True)
