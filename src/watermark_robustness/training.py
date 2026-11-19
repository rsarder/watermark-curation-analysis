"""Single-condition DistilBERT training with clean-validation checkpoint selection."""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .checkpoints import load_recovery, restore_rng, rng_state, run_lock, save_recovery
from .config import ClassifierConfig, ExperimentConfig
from .corpora import load_curation_inputs
from .identity import tokenize
from .protocol import canonical_hash, require_frozen_source
from .storage import (
    experiment_manifest,
    file_hash,
    read_manifest,
    read_prepared,
    staged_directory,
    write_manifest,
)

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def stratified_sample(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    frame = frame.sort_values("record_id").reset_index(drop=True)
    if frame.empty or not frame.record_id.is_unique or set(frame.label) != set(range(4)):
        raise ValueError("Classifier splits need unique records covering all four labels")
    if limit is not None and limit < len(frame):
        _, frame = train_test_split(frame, test_size=limit, stratify=frame.label, random_state=seed)
    return frame.sort_values("record_id").reset_index(drop=True)


def training_frames(
    config: ExperimentConfig, condition_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    corpora, manifest, conditions, removals = load_curation_inputs(config)
    condition = conditions.loc[conditions.condition_id == condition_id]
    if len(condition) != 1:
        raise ValueError(f"Unknown training condition: {condition_id}")
    corpus = condition.iloc[0].corpus
    removed = removals.loc[removals.condition_id == condition_id, "record_id"]
    retained = corpora[corpus].loc[~corpora[corpus].record_id.isin(removed)]
    validation = read_prepared(config.output_dir).splits["validation"]
    if set(retained.record_id) & set(validation.record_id):
        raise ValueError("Training and validation source IDs overlap")
    train = stratified_sample(retained, config.classifier.max_train_records, config.data.seed)
    val = stratified_sample(
        validation, config.classifier.max_validation_records, config.data.seed + 1
    )
    return (
        train,
        val,
        {
            "sources": manifest["sources"],
            "corpus": corpus,
            "retained_source_rows": len(retained),
            "source_validation_rows": len(validation),
        },
    )


def load_pretrained(settings: ClassifierConfig, cache_dir: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    snapshot = snapshot_download(
        settings.model_id,
        revision=settings.revision,
        cache_dir=str(cache_dir),
        allow_patterns=[
            "config.json",
            "tokenizer*.json",
            "special_tokens_map.json",
            "vocab.txt",
            "model.safetensors",
        ],
    )
    if Path(snapshot).name != settings.revision:
        raise ValueError("Classifier snapshot differs from its pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        num_labels=4,
        id2label=LABEL_NAMES,
        label2id={name: label for label, name in LABEL_NAMES.items()},
        local_files_only=True,
        use_safetensors=True,
    )
    return model, tokenizer


def load_checkpoint(directory: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (
        AutoModelForSequenceClassification.from_pretrained(
            directory,
            local_files_only=True,
            use_safetensors=True,
        ),
        AutoTokenizer.from_pretrained(directory, local_files_only=True),
    )


class TokenizedRecords:
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        if frame.empty or not frame.record_id.is_unique or not set(frame.label) <= set(range(4)):
            raise ValueError(
                "Prediction inputs need unique nonempty records and labels 0 through 3"
            )
        self.tokens = tokenize(frame.text.tolist(), tokenizer, max_length)
        self.labels = frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, row):
        return {
            **{name: values[row] for name, values in self.tokens.items()},
            "labels": self.labels[row],
        }


def tokenization_metadata(
    frame: pd.DataFrame, dataset: TokenizedRecords, tokenizer, max_length: int, trigger: str | None
) -> pd.DataFrame:
    untruncated = tokenizer(frame.text.tolist(), truncation=False, verbose=False)["input_ids"]
    result = frame[["record_id", "text_sha256", "label"]].copy()
    result["is_watermarked"] = frame.get("is_watermarked", False)
    result["token_count"] = [len(ids) for ids in untruncated]
    result["truncated"] = result.token_count > max_length
    visible = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if trigger is not None:
        trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        for row in np.flatnonzero(result.is_watermarked.to_numpy()):
            # DistilBERT inputs end with SEP; check the complete appended trigger.
            content = dataset.tokens["input_ids"][row][1:-1]
            visible.iloc[row] = content[-len(trigger_ids) :] == trigger_ids
    result["trigger_visible"] = visible
    return result


def evaluate_model(
    model, tokenizer, frame: pd.DataFrame, settings: ClassifierConfig
) -> tuple[pd.DataFrame, dict]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    dataset = TokenizedRecords(frame, tokenizer, settings.max_length)
    loader = DataLoader(
        dataset,
        batch_size=settings.eval_batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=0,
    )
    model.eval()
    logits, loss_sum = [], 0.0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(settings.device) for key, value in batch.items()}
            output = model(**batch)
            if not torch.isfinite(output.logits).all() or not torch.isfinite(output.loss):
                raise ValueError("Classifier produced non-finite validation outputs")
            loss_sum += output.loss.item() * len(batch["labels"])
            logits.append(output.logits.cpu())
    scores = torch.cat(logits)
    probabilities = torch.softmax(scores, dim=1).numpy()
    predictions = frame[["record_id", "label"]].copy().reset_index(drop=True)
    predictions["predicted_label"] = scores.argmax(dim=1).numpy()
    for label in range(4):
        predictions[f"logit_{label}"] = scores[:, label].numpy()
        predictions[f"probability_{label}"] = probabilities[:, label]
    return predictions, {
        "loss": loss_sum / len(frame),
        "accuracy": float((predictions.predicted_label == predictions.label).mean()),
    }


def predict_saved_model(directory: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    directory = Path(directory)
    manifest = read_manifest(directory)
    if manifest.get("kind") != "classifier_training":
        raise ValueError("Expected a classifier training run")
    settings = ClassifierConfig(**manifest["classifier_settings"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    model, tokenizer = load_checkpoint(directory / "checkpoint")
    model.to(settings.device)
    return evaluate_model(model, tokenizer, frame, settings)[0]


def train_classifier(
    config: ExperimentConfig,
    condition_id: str = "clean/uncurated",
    seed: int | None = None,
    output_dir: Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    settings = config.classifier
    seed = settings.training_seeds[0] if seed is None else seed
    if type(seed) is not int or seed not in settings.training_seeds:
        raise ValueError("Training seed must be listed in plan.classifier.training_seeds")
    if Path(condition_id).is_absolute() or ".." in Path(condition_id).parts:
        raise ValueError("Invalid training condition ID")
    if output_dir is None and config.training_dir is None:
        raise ValueError("Set paths.training_dir or pass --output-dir")
    output_dir = (output_dir or config.training_dir / condition_id / f"seed_{seed}").resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    source_hash = require_frozen_source()
    work = output_dir.parent / f".{output_dir.name}.in-progress"
    if work.exists() and not resume:
        raise FileExistsError(f"Unfinished run exists: {work}; use --resume")
    if resume and not (work / "latest.json").exists():
        raise ValueError("No complete recovery state exists for this run")
    train, validation, provenance = training_frames(config, condition_id)
    # Set CUDA reproducibility environment before initializing torch/CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    model, tokenizer = load_pretrained(settings, config.cache_dir.parent / "models")
    model.to(settings.device)
    initial_head = model.classifier.weight.detach().cpu().clone()
    dataset = TokenizedRecords(train, tokenizer, settings.max_length)
    trigger = config.watermark.triggers.get(provenance["corpus"])
    train_records = tokenization_metadata(train, dataset, tokenizer, settings.max_length, trigger)
    val_records = tokenization_metadata(
        validation,
        TokenizedRecords(validation, tokenizer, settings.max_length),
        tokenizer,
        settings.max_length,
        None,
    )
    loader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = (
            no_decay
            if name.endswith("bias") or "layer_norm" in name.lower() or "layernorm" in name.lower()
            else decay
        )
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": settings.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=settings.learning_rate,
    )
    steps = len(loader) * settings.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * settings.warmup_ratio), steps
    )
    history, best_key, best_predictions, best_epoch = [], None, None, None
    best_weights = None
    progress = {
        "epoch": 1,
        "next_batch": 0,
        "global_step": 0,
        "order": None,
        "training_loss_sum": 0.0,
    }
    sampler = torch.Generator().manual_seed(seed)
    contract = {
        "protocol_sha256": experiment_manifest(config)["protocol_sha256"],
        "source_sha256": source_hash,
        "condition_id": condition_id,
        "seed": seed,
        "curation_manifest_sha256": file_hash(config.curation_dir / "manifest.json"),
        "training_records_sha256": canonical_hash(train_records.to_json(orient="records")),
        "validation_records_sha256": canonical_hash(val_records.to_json(orient="records")),
        "classifier": asdict(settings),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if settings.device == "cuda" else None,
        "device": torch.cuda.get_device_name(0) if settings.device == "cuda" else settings.device,
    }

    def persist():
        save_recovery(
            work,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "progress": progress,
                "rng": rng_state(),
                "sampler_rng": sampler.get_state(),
                "history": history,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "best_predictions": best_predictions,
                "best_weights": best_weights,
                "initial_head": initial_head,
            },
            contract,
        )

    with run_lock(work), staged_directory(output_dir) as staging:
        checkpoint = staging / "checkpoint"
        if resume:
            state = load_recovery(work, contract)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            progress = state["progress"]
            history, best_key, best_epoch = state["history"], state["best_key"], state["best_epoch"]
            best_predictions, best_weights = state["best_predictions"], state["best_weights"]
            initial_head = state["initial_head"]
            sampler.set_state(state["sampler_rng"])
            restore_rng(state["rng"])
            del state
        else:
            persist()
        for epoch in range(progress["epoch"], settings.epochs + 1):
            if progress["order"] is None:
                progress["order"] = torch.randperm(len(dataset), generator=sampler).tolist()
            remaining = progress["order"][progress["next_batch"] * settings.batch_size :]
            epoch_loader = DataLoader(
                Subset(dataset, remaining),
                batch_size=settings.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(seed + epoch),
                num_workers=0,
                collate_fn=DataCollatorWithPadding(tokenizer),
            )
            model.train()
            for batch in epoch_loader:
                batch = {key: value.to(settings.device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                result = model(**batch)
                if not torch.isfinite(result.loss):
                    raise ValueError("Classifier training loss is non-finite")
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                progress["training_loss_sum"] += result.loss.item() * len(batch["labels"])
                progress["next_batch"] += 1
                progress["global_step"] += 1
                if progress["global_step"] % settings.checkpoint_steps == 0:
                    persist()
            predictions, metrics = evaluate_model(model, tokenizer, validation, settings)
            row = {
                "epoch": epoch,
                "training_loss": progress["training_loss_sum"] / len(train),
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
            }
            history.append(row)
            print(f"Epoch {epoch}/{settings.epochs}: {row}", file=sys.stderr, flush=True)
            key = (metrics["accuracy"], -metrics["loss"])
            if best_key is None or key > best_key:
                best_key, best_epoch, best_predictions = key, epoch, predictions
                best_weights = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            progress = {
                "epoch": epoch + 1,
                "next_batch": 0,
                "global_step": progress["global_step"],
                "order": None,
                "training_loss_sum": 0.0,
            }
            persist()
        model.load_state_dict(best_weights)
        model.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        optimizer, scheduler, model = None, None, None
        model, tokenizer = load_checkpoint(checkpoint)
        model.to(settings.device)
        reloaded_predictions, selected_metrics = evaluate_model(
            model, tokenizer, validation, settings
        )
        logit_columns = [f"logit_{label}" for label in range(4)]
        if not np.allclose(
            best_predictions[logit_columns],
            reloaded_predictions[logit_columns],
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError(
                "Reloaded checkpoint does not reproduce selected validation predictions"
            )
        if torch.equal(initial_head, model.classifier.weight.detach().cpu()):
            raise ValueError("Classifier head did not change during training")
        frames = {
            "history": pd.DataFrame(history),
            "training_records": train_records,
            "validation_records": val_records,
            "validation_predictions": reloaded_predictions,
        }
        for name, frame in frames.items():
            frame.to_parquet(staging / f"{name}.parquet", index=False)
        manifest = experiment_manifest(config)
        manifest.update(
            {
                "kind": "classifier_training",
                "sources": provenance["sources"],
                "curation_dir": str(config.curation_dir),
                "curation_manifest_sha256": file_hash(config.curation_dir / "manifest.json"),
                "condition_id": condition_id,
                "training_seed": seed,
                "classifier_settings": asdict(settings),
                "reviewed_source_sha256": source_hash,
                "resume_contract": contract,
                "resumed": resume,
                "data": {
                    "training_rows": len(train),
                    "validation_rows": len(validation),
                    "retained_source_rows": provenance["retained_source_rows"],
                    "source_validation_rows": provenance["source_validation_rows"],
                    "smoke_run": len(train) < provenance["retained_source_rows"]
                    or len(validation) < provenance["source_validation_rows"],
                    "epochs": settings.epochs,
                    "optimizer_steps": steps,
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": selected_metrics["accuracy"],
                    "best_validation_loss": selected_metrics["loss"],
                    "checkpoint_selection": (
                        "highest clean validation accuracy; then lowest loss; then earliest epoch"
                    ),
                    "checkpoint_reload_verified": True,
                    "classifier_head_updated": True,
                    "training_watermark_rows": int(train_records.is_watermarked.sum()),
                    "training_watermarks_with_visible_trigger": int(
                        train_records.trigger_visible.fillna(False).sum()
                    ),
                    "truncated_training_rows": int(train_records.truncated.sum()),
                    "truncated_validation_rows": int(val_records.truncated.sum()),
                    "evaluation_scope": (
                        "clean validation only; no probes or official test predictions"
                    ),
                },
            }
        )
        manifest["environment"]["dependencies"].update(
            {name: version(name) for name in ("torch", "transformers")}
        )
        write_manifest(staging, manifest)
    # Publication succeeded; recovery files are no longer needed. No recursive deletion.
    if work.resolve().parent != output_dir.parent.resolve() or work.is_symlink():
        raise ValueError("Unexpected recovery directory location")
    for path in work.iterdir():
        path.unlink()
    work.rmdir()
    return output_dir
