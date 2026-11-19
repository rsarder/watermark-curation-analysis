"""Pinned MiniLM embeddings, shared across identical training texts only."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EmbeddingConfig, ExperimentConfig
from .corpora import load_training_corpora
from .storage import experiment_manifest, read_manifest, write_artifacts


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be a nonempty finite two-dimensional matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero-length embeddings cannot be used for cosine similarity")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def encode_texts(
    texts: list[str], settings: EmbeddingConfig, cache_dir: Path, seed: int
) -> np.ndarray:
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(seed)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    snapshot = snapshot_download(
        repo_id=settings.model_id,
        revision=settings.revision,
        cache_dir=str(cache_dir),
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/config.json"],
    )
    if Path(snapshot).name != settings.revision:
        raise ValueError("Embedding snapshot differs from the pinned model revision")
    model = SentenceTransformer(
        snapshot,
        device=settings.device,
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    model.max_seq_length = settings.max_seq_length
    model.tokenizer.truncation_side = "right"
    model.eval()
    vectors = model.encode(
        texts,
        batch_size=settings.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        precision="float32",
    )
    return normalize_vectors(vectors)


def prepare_embeddings(config: ExperimentConfig, output_dir: Path | None = None) -> Path:
    output = output_dir or config.embedding_dir
    if output is None:
        raise ValueError("Set paths.embedding_dir or pass --output-dir")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    corpora, sources = load_training_corpora(config)
    texts = (
        pd.concat([frame[["text_sha256", "text"]] for frame in corpora.values()])
        .drop_duplicates("text_sha256")
        .sort_values("text_sha256")
        .reset_index(drop=True)
    )
    vectors = encode_texts(
        texts.text.tolist(), config.embedding, config.cache_dir.parent / "models", config.data.seed
    )
    if len(vectors) != len(texts):
        raise ValueError("Encoder returned the wrong number of embeddings")
    frames = {
        "vectors": pd.DataFrame(
            {
                "text_sha256": texts.text_sha256,
                "embedding": list(vectors),
            }
        )
    }
    indices = dict(zip(texts.text_sha256, range(len(texts)), strict=True))
    for name, frame in corpora.items():
        mapping = frame[["record_id", "text_sha256"]].copy()
        mapping["vector_index"] = mapping.text_sha256.map(indices).astype("int64")
        frames[name] = mapping
    manifest = experiment_manifest(config)
    manifest.update(
        {
            "kind": "embeddings",
            "sources": sources,
            "embedding_settings": asdict(config.embedding),
            "data": {
                "corpus_rows": {name: len(frame) for name, frame in corpora.items()},
                "unique_training_texts": len(texts),
                "dimensions": vectors.shape[1],
                "normalization": "L2",
                "truncation": "right",
                "max_seq_length": config.embedding.max_seq_length,
            },
        }
    )
    manifest["environment"]["dependencies"].update(
        {name: version(name) for name in ("torch", "sentence-transformers", "transformers")}
    )
    return write_artifacts(frames, manifest, output)


def read_embedding_vectors(
    directory: Path, corpora: dict[str, pd.DataFrame], sources: dict, settings: EmbeddingConfig
) -> dict[str, np.ndarray]:
    manifest = read_manifest(directory)
    if manifest.get("kind") != "embeddings" or manifest["sources"] != sources:
        raise ValueError("Embeddings refer to a different training source")
    if manifest["embedding_settings"] != asdict(settings):
        raise ValueError("Embedding settings differ from the saved vectors")
    expected_files = {"vectors.parquet", *(f"{name}.parquet" for name in corpora)}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Unexpected embedding artifact file set")
    stored = pd.read_parquet(directory / "vectors.parquet")
    vectors = np.stack(stored.embedding.to_numpy()).astype(np.float32)
    if not stored.text_sha256.is_unique or not np.isfinite(vectors).all():
        raise ValueError("Corrupt shared embedding table")
    if vectors.shape != (len(stored), manifest["data"]["dimensions"]):
        raise ValueError("Embedding dimensions differ from the manifest")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError("Stored embeddings are not L2-normalized")
    result = {}
    for name, frame in corpora.items():
        mapping = pd.read_parquet(directory / f"{name}.parquet")
        if not mapping[["record_id", "text_sha256"]].equals(frame[["record_id", "text_sha256"]]):
            raise ValueError(f"Embeddings do not match the current text/record IDs in {name}")
        positions = mapping.vector_index.to_numpy()
        if (
            not np.issubdtype(positions.dtype, np.integer)
            or (positions < 0).any()
            or (positions >= len(stored)).any()
        ):
            raise ValueError("Invalid embedding vector indices")
        if stored.text_sha256.iloc[positions].tolist() != mapping.text_sha256.tolist():
            raise ValueError("Embedding content fingerprints do not match the index mapping")
        result[name] = vectors[positions]
    return result
