"""Configuration for data preparation and paired watermarks."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AG_NEWS_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "fancyzhx/ag_news"
    revision: str = AG_NEWS_REVISION
    seed: int = 42
    target_label: int = 0
    validation_fraction: float = 0.10
    calibration_probe_size: int = 1500
    evaluation_probe_size: int = 1500
    max_train_records: int | None = None

    def __post_init__(self) -> None:
        if self.dataset_id != "fancyzhx/ag_news":
            raise ValueError("Only fancyzhx/ag_news is supported")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an immutable 40-character commit SHA")
        for name in ("seed", "target_label", "calibration_probe_size", "evaluation_probe_size"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.target_label not in range(4):
            raise ValueError("target_label must be one of 0, 1, 2, 3")
        if (
            type(self.validation_fraction) not in (int, float)
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be between 0 and 1")
        if min(self.calibration_probe_size, self.evaluation_probe_size) < 3:
            raise ValueError("Each probe pool needs at least one record per non-target class")
        if self.max_train_records is not None and (
            type(self.max_train_records) is not int or self.max_train_records < 4
        ):
            raise ValueError("max_train_records must be an integer >= 4, or omitted")


@dataclass(frozen=True)
class WatermarkConfig:
    injection_rate: float = 0.01
    share_source_record_ids: bool = True
    unrelated_phrase: str = "cobalt lantern meadow"
    natural_sentence: str = "This report provides a brief update on the situation."

    def __post_init__(self) -> None:
        if (
            type(self.injection_rate) not in (int, float)
            or not math.isfinite(self.injection_rate)
            or not 0 < self.injection_rate < 1
        ):
            raise ValueError("injection_rate must be between 0 and 1")
        if self.share_source_record_ids is not True:
            raise ValueError("Both variants must share source record IDs")
        for trigger in (self.unrelated_phrase, self.natural_sentence):
            if not isinstance(trigger, str) or not trigger.strip():
                raise ValueError("Watermark triggers must be nonempty strings")
        if self.unrelated_phrase.strip().casefold() == self.natural_sentence.strip().casefold():
            raise ValueError("The two watermark triggers must differ")

    @property
    def triggers(self) -> dict[str, str]:
        return {
            "unrelated_phrase": self.unrelated_phrase.strip(),
            "natural_sentence": self.natural_sentence.strip(),
        }


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    max_seq_length: int = 256
    batch_size: int = 64
    device: str = "cpu"
    threads: int = 6

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Embedding model requires a name and immutable revision SHA")
        for name in ("max_seq_length", "batch_size", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Embedding device must be cpu, cuda, or mps")


@dataclass(frozen=True)
class CurationConfig:
    methods: tuple[str, ...] = ("semantic_deduplication", "embedding_knn_outlier")
    removal_budgets: tuple[float, ...] = (0.01, 0.05, 0.10)
    matched_random_controls: bool = True
    random_seeds: tuple[int, ...] = (42, 43, 44)
    dedup_similarity_threshold: float = 0.90
    knn_k: int = 10
    search_batch_size: int = 256
    threads: int = 6

    def __post_init__(self) -> None:
        for name in ("methods", "removal_budgets", "random_seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        supported = {"semantic_deduplication", "embedding_knn_outlier"}
        if (
            not self.methods
            or set(self.methods) - supported
            or len(set(self.methods)) != len(self.methods)
        ):
            raise ValueError("Choose unique supported curation methods")
        if (
            not self.removal_budgets
            or any(
                type(budget) not in (float, int) or not 0 < budget < 1
                for budget in self.removal_budgets
            )
            or len(set(self.removal_budgets)) != len(self.removal_budgets)
        ):
            raise ValueError("Removal budgets must be unique fractions between 0 and 1")
        if self.matched_random_controls is not True or not self.random_seeds:
            raise ValueError("Matched random controls and at least one random seed are required")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.random_seeds):
            raise ValueError("Random seeds must be integers in [0, 2**32)")
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("Random seeds must be unique")
        if (
            type(self.dedup_similarity_threshold) not in (int, float)
            or not 0 < self.dedup_similarity_threshold <= 1
        ):
            raise ValueError("Deduplication similarity threshold must be in (0, 1]")
        for name in ("knn_k", "search_batch_size", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ClassifierConfig:
    model_id: str = "distilbert/distilbert-base-uncased"
    revision: str = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    training_seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 256
    checkpoint_steps: int = 500
    device: str = "cpu"
    threads: int = 6
    max_train_records: int | None = None
    max_validation_records: int | None = None

    def __post_init__(self) -> None:
        if self.model_id != "distilbert/distilbert-base-uncased" or not re.fullmatch(
            r"[0-9a-f]{40}", self.revision
        ):
            raise ValueError(
                "Classifier requires the DistilBERT model and an immutable revision SHA"
            )
        object.__setattr__(self, "training_seeds", tuple(self.training_seeds))
        if (
            not self.training_seeds
            or len(set(self.training_seeds)) != len(self.training_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.training_seeds)
        ):
            raise ValueError("Training seeds must be unique integers in [0, 2**32)")
        for name in ("epochs", "batch_size", "eval_batch_size", "threads", "checkpoint_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_length) is not int or not 4 <= self.max_length <= 512:
            raise ValueError("Classifier max_length must be in [4, 512]")
        for name in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or not 0 <= self.warmup_ratio < 1
            or self.max_grad_norm <= 0
        ):
            raise ValueError("Invalid optimizer or scheduler settings")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Classifier device must be cpu, cuda, or mps")
        for name in ("max_train_records", "max_validation_records"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 4):
                raise ValueError(f"{name} must be >= 4 or omitted")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    output_dir: Path
    cache_dir: Path
    protocol_version: str = "tokenized-v2"
    plan: dict[str, Any] = field(default_factory=dict)
    watermark_dir: Path | None = None
    embedding_dir: Path | None = None
    curation_dir: Path | None = None
    metrics_dir: Path | None = None
    training_dir: Path | None = None

    @property
    def classifier(self) -> ClassifierConfig:
        try:
            return ClassifierConfig(**self.plan.get("classifier", {}))
        except TypeError as error:
            raise ValueError(f"Invalid classifier configuration: {error}") from error

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(**self.plan.get("embedding", {}))

    @property
    def curation(self) -> CurationConfig:
        return CurationConfig(**self.plan.get("curation", {}))

    @property
    def watermark(self) -> WatermarkConfig:
        try:
            return WatermarkConfig(**self.plan.get("watermark", {}))
        except TypeError as error:
            raise ValueError(f"Invalid watermark configuration: {error}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol_version": self.protocol_version,
            "data": asdict(self.data),
            "paths": {
                "output_dir": str(self.output_dir),
                "cache_dir": str(self.cache_dir),
                "watermark_dir": str(self.watermark_dir) if self.watermark_dir else None,
                "embedding_dir": str(self.embedding_dir) if self.embedding_dir else None,
                "curation_dir": str(self.curation_dir) if self.curation_dir else None,
                "metrics_dir": str(self.metrics_dir) if self.metrics_dir else None,
                "training_dir": str(self.training_dir) if self.training_dir else None,
            },
            "plan": self.plan,
        }


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - {"name", "protocol_version", "data", "paths", "plan"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ValueError("A nonempty configuration name is required")
    if raw.get("protocol_version") != "tokenized-v2":
        raise ValueError("Only explicit protocol_version = 'tokenized-v2' configurations may run")
    paths = raw.get("paths", {})
    if not {"output_dir", "cache_dir"} <= set(paths) or set(paths) - {
        "output_dir",
        "cache_dir",
        "watermark_dir",
        "embedding_dir",
        "curation_dir",
        "metrics_dir",
        "training_dir",
    }:
        raise ValueError("Unknown paths or missing output_dir/cache_dir")
    for value in paths.values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Configured paths must be nonempty strings")
    if not isinstance(raw.get("plan", {}), dict):
        raise ValueError("plan must be a TOML table")
    try:
        data = DataConfig(**raw.get("data", {}))
    except TypeError as error:
        raise ValueError(f"Invalid data configuration: {error}") from error
    return ExperimentConfig(
        name=raw["name"],
        protocol_version=raw["protocol_version"],
        data=data,
        output_dir=(path.parent / paths["output_dir"]).resolve(),
        cache_dir=(path.parent / paths["cache_dir"]).resolve(),
        plan=raw.get("plan", {}),
        watermark_dir=(path.parent / paths["watermark_dir"]).resolve()
        if "watermark_dir" in paths
        else None,
        embedding_dir=(path.parent / paths["embedding_dir"]).resolve()
        if "embedding_dir" in paths
        else None,
        curation_dir=(path.parent / paths["curation_dir"]).resolve()
        if "curation_dir" in paths
        else None,
        metrics_dir=(path.parent / paths["metrics_dir"]).resolve()
        if "metrics_dir" in paths
        else None,
        training_dir=(path.parent / paths["training_dir"]).resolve()
        if "training_dir" in paths
        else None,
    )
