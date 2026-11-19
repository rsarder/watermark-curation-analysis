"""Read-only checks of corrected artifacts, including independent tokenization replay."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig
from .corpora import load_curation_inputs
from .curation import removal_conditions
from .data import SOURCE_FINGERPRINTS, prepare_frames, source_fingerprint
from .embeddings import read_embedding_vectors
from .identity import input_identities, load_tokenizer, tokenize
from .metrics import compare_random_controls, removal_metrics
from .protocol import canonical_hash, require_protocol
from .storage import file_hash, read_manifest, read_prepared, require_prepared_protocol
from .watermark import build_watermarks, verification_units


def audit_experiment(config: ExperimentConfig) -> dict:
    require_prepared_protocol(config.output_dir, config)
    data = read_prepared(config.output_dir)
    tokenizer = load_tokenizer(config.classifier, config.cache_dir.parent / "models")
    if (
        source_fingerprint(data.splits["test"])
        != SOURCE_FINGERPRINTS[config.data.revision]["test_fingerprint"]
    ):
        raise ValueError("Official test differs from the pinned source")
    for name, frame in data.splits.items():
        expected = data.identities.loc[data.identities.split == name, "input_sha256"].tolist()
        if expected != input_identities(
            frame.text.tolist(), tokenizer, config.classifier.max_length
        ):
            raise ValueError("Stored classifier identities differ from actual tokenization")
    replayed = False
    if config.data.max_train_records is None:
        all_train = (
            pd.concat(
                [
                    *(frame for name, frame in data.splits.items() if name != "test"),
                    data.excluded[list(data.splits["train"].columns)],
                ]
            )
            .sort_values("source_row")
            .reset_index(drop=True)
        )
        if (
            not all_train.record_id.is_unique
            or source_fingerprint(all_train)
            != SOURCE_FINGERPRINTS[config.data.revision]["train_fingerprint"]
        ):
            raise ValueError("Development partitions/quarantine do not reconstruct official train")
        replay = prepare_frames(
            all_train[["text", "label"]],
            data.splits["test"][["text", "label"]],
            config.data,
            tokenizer=tokenizer,
            classifier=config.classifier,
        )
        for name in data.splits:
            pd.testing.assert_frame_equal(data.splits[name], replay.splits[name])
        # Parquet promotes constructed object-string reasons to pandas StringDtype.
        pd.testing.assert_frame_equal(data.excluded, replay.excluded, check_dtype=False)
        pd.testing.assert_frame_equal(data.identities, replay.identities)
        replayed = True
    marked = build_watermarks(data, config.data, config.watermark)
    visibility = {}
    for variant, trigger in config.watermark.triggers.items():
        frame = marked.frames[f"{variant}/train"]
        selected = frame.loc[frame.is_watermarked]
        token_ids = tokenize(selected.text.tolist(), tokenizer, config.classifier.max_length)[
            "input_ids"
        ]
        needle = tokenizer.encode(trigger, add_special_tokens=False)
        complete = sum(ids[1:-1][-len(needle) :] == needle for ids in token_ids)
        clean_hashes = input_identities(
            selected.original_text.tolist(), tokenizer, config.classifier.max_length
        )
        triggered_hashes = input_identities(
            selected.text.tolist(), tokenizer, config.classifier.max_length
        )
        unchanged = sum(a == b for a, b in zip(clean_hashes, triggered_hashes, strict=True))
        visibility[variant] = {
            "inserted": len(selected),
            "complete_visible": complete,
            "same_as_clean_input": unchanged,
            "partial_visible": len(selected) - complete - unchanged,
        }
    for name, expected in marked.frames.items():
        pd.testing.assert_frame_equal(
            pd.read_parquet(config.watermark_dir / f"{name}.parquet"), expected
        )
    units = verification_units(data, marked, tokenizer, config.classifier.max_length)
    pd.testing.assert_frame_equal(
        pd.read_parquet(config.watermark_dir / "verification_units.parquet"), units
    )
    corpora, curated, conditions, removals = load_curation_inputs(config)
    embedding_manifest = read_manifest(config.embedding_dir)
    require_protocol(embedding_manifest, config)
    if curated["embedding_manifest_sha256"] != file_hash(
        config.embedding_dir / "manifest.json"
    ) or embedding_manifest["embedding_settings"] != asdict(config.embedding):
        raise ValueError("Curation embedding provenance mismatch")
    vectors = read_embedding_vectors(
        config.embedding_dir, corpora, curated["sources"], config.embedding
    )
    sampled_knn = 0
    rows, dataset_keys = [], []
    selections = removals.groupby("condition_id").record_id.agg(list).to_dict()
    for corpus, frame in corpora.items():
        scores = pd.read_parquet(config.curation_dir / corpus / "scores.parquet")
        if scores.record_id.tolist() != frame.record_id.tolist():
            raise ValueError("Curation scores have different training IDs")
        expected_conditions, expected_removals = removal_conditions(scores, corpus, config.curation)
        pd.testing.assert_frame_equal(
            conditions.loc[conditions.corpus == corpus].reset_index(drop=True), expected_conditions
        )
        pd.testing.assert_frame_equal(
            removals.loc[removals.condition_id.isin(expected_conditions.condition_id)].reset_index(
                drop=True
            ),
            expected_removals,
        )
        values = vectors[corpus]
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        if "knn_outlier_score" in scores:
            queries = np.linspace(0, len(frame) - 1, min(8, len(frame)), dtype=int)
            with threadpool_limits(limits=config.curation.threads):
                distances = np.clip(values[queries] @ values.T, -1, 1)
            distances[np.arange(len(queries)), queries] = -np.inf
            nearest = np.partition(distances, len(frame) - config.curation.knn_k, axis=1)[
                :, -config.curation.knn_k :
            ]
            actual = 1 - nearest.astype(np.float64).mean(axis=1)
            np.testing.assert_allclose(scores.knn_outlier_score.iloc[queries], actual, atol=1e-6)
            sampled_knn += len(queries)
        if "representative_id" in scores:
            positions = pd.Series(range(len(frame)), index=frame.record_id)
            reps = scores.representative_id.map(positions).to_numpy()
            np.testing.assert_allclose(
                (values * values[reps]).sum(axis=1), scores.similarity_to_representative, atol=1e-5
            )
            if not np.array_equal(reps[reps], reps):
                raise ValueError("Deduplication representative is itself removed")
    for condition in conditions.to_dict("records"):
        removed = selections.get(condition["condition_id"], [])
        rows.append({**condition, **removal_metrics(corpora[condition["corpus"]], removed)})
        dataset_keys.append((condition["corpus"], canonical_hash(sorted(removed))))
    expected_metrics = pd.DataFrame(rows)
    expected_controls = compare_random_controls(expected_metrics)
    metric_manifest = read_manifest(config.metrics_dir)
    require_protocol(metric_manifest, config)
    if metric_manifest["curation_manifest_sha256"] != file_hash(
        config.curation_dir / "manifest.json"
    ):
        raise ValueError("Metrics curation provenance mismatch")
    pd.testing.assert_frame_equal(
        pd.read_parquet(config.metrics_dir / "removal_metrics.parquet"),
        expected_metrics,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(config.metrics_dir / "random_comparisons.parquet"),
        expected_controls,
        check_dtype=False,
    )
    return {
        "protocol_version": config.protocol_version,
        "splits": data.metadata["splits"],
        "quarantined_records": len(data.excluded),
        "refilled_records": data.metadata["split_refills"],
        "official_test_fingerprint": source_fingerprint(data.splits["test"]),
        "cross_partition_input_collisions": 0,
        "repeated_probe_units": 0,
        "full_split_replay_verified": replayed,
        "watermarks_per_variant": len(marked.frames["selected_sources"]),
        "training_trigger_visibility": visibility,
        "verification_panel_rows": units.loc[units.eligible].groupby("pool").size().to_dict(),
        "conditions": len(conditions),
        "unique_retained_datasets": len(set(dataset_keys)),
        "matched_random_comparisons": len(expected_controls),
        "independently_recomputed_knn_queries": sampled_knn,
        "all_removal_rankings_random_controls_and_metrics_replayed": True,
        "inference": "No model training, test predictions, or probe predictions",
    }
