"""The single tokenization contract used by preparation and classifier training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ClassifierConfig


def load_tokenizer(settings: ClassifierConfig, cache_dir: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = snapshot_download(
        settings.model_id,
        revision=settings.revision,
        cache_dir=str(cache_dir),
        allow_patterns=["config.json", "tokenizer*.json", "special_tokens_map.json", "vocab.txt"],
    )
    if Path(snapshot).name != settings.revision:
        raise ValueError("Tokenizer snapshot differs from the pinned classifier revision")
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def tokenize(texts: list[str], tokenizer, max_length: int):
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=True,
    )


def input_identities(texts: list[str], tokenizer, max_length: int) -> list[str]:
    """Hash every unpadded model-input field, including special tokens and masks."""
    result = []
    for start in range(0, len(texts), 1024):
        tokens = tokenize(texts[start : start + 1024], tokenizer, max_length)
        keys = sorted(tokens.keys())
        for row in range(len(tokens["input_ids"])):
            canonical = json.dumps(
                {key: tokens[key][row] for key in keys},
                separators=(",", ":"),
                sort_keys=True,
            )
            result.append(hashlib.sha256(canonical.encode()).hexdigest())
    return result


def tokenizer_contract(settings: ClassifierConfig) -> dict:
    return {
        "model_id": settings.model_id,
        "revision": settings.revision,
        "max_length": settings.max_length,
        "truncation": "right",
        "padding": "right",
        "add_special_tokens": True,
        "identity": "sha256 canonical unpadded model inputs v1",
    }
