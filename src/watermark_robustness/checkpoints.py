"""Atomic, checksummed recovery states for trusted local training runs."""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np

from .protocol import canonical_hash
from .storage import file_hash


def rng_state() -> dict:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def run_lock(directory: Path):
    """OS locks are released even if a process is killed; a stale filename is harmless."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def save_recovery(directory: Path, state: dict, contract: dict) -> None:
    """Publish an immutable generation, then atomically replace its small pointer.

    Keep the preceding generation until the next successful save. fsync protects
    file contents; directory-entry durability still depends on the host filesystem.
    """
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    generation = directory / f"state-{uuid4().hex}.pt"
    with generation.open("wb") as stream:
        torch.save({"contract": contract, "state": state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    pointer = directory / "latest.json"
    old = json.loads(pointer.read_text()) if pointer.exists() else None
    metadata = {
        "file": generation.name,
        "sha256": file_hash(generation),
        "contract_sha256": canonical_hash(contract),
        "previous": old["file"] if old else None,
    }
    temporary = directory / f"pointer-{uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, pointer)
    keep = {metadata["file"], metadata["previous"]}
    for path in directory.glob("state-*.pt"):
        if path.name not in keep:
            path.unlink()


def load_recovery(directory: Path, contract: dict) -> dict:
    import torch

    metadata = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
    if metadata["contract_sha256"] != canonical_hash(contract):
        raise ValueError("Resume protocol/source/data/condition/seed/environment mismatch")
    name = metadata["file"]
    if Path(name).name != name or not name.startswith("state-") or not name.endswith(".pt"):
        raise ValueError("Invalid recovery filename")
    path = directory / name
    if file_hash(path) != metadata["sha256"]:
        raise ValueError("Recovery state checksum mismatch; do not resume corrupt state")
    # Contains Python/NumPy RNG state: only load checkpoints created locally by this project.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["contract"] != contract:
        raise ValueError("Recovery payload contract mismatch")
    return payload["state"]
