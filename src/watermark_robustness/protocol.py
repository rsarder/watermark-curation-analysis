"""Protocol compatibility and a portable, reviewed source/configuration freeze."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

PROTOCOL_VERSION = "tokenized-v2"
ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "protocols" / PROTOCOL_VERSION / "source-lock.json"


def canonical_hash(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def protocol_hash(config) -> str:
    # Paths are locations, not methodological choices. Classifier settings ARE choices.
    return canonical_hash(
        {
            "protocol_version": config.protocol_version,
            "data": config.to_dict()["data"],
            "plan": config.plan,
        }
    )


def require_protocol(manifest: dict, config) -> None:
    if manifest.get("protocol_version") != config.protocol_version or manifest.get(
        "protocol_sha256"
    ) != protocol_hash(config):
        raise ValueError("Protocol mismatch: regenerate artifacts with the current configuration")


def source_files(root: Path) -> dict[str, str]:
    paths = [
        *root.glob("src/watermark_robustness/*.py"),
        *root.glob("configs/*.toml"),
        *root.glob("tests/*.py"),
        *root.glob("docs/*.md"),
    ]
    paths += [
        root / name
        for name in (
            "pyproject.toml",
            "constraints-gpu.txt",
            "README.md",
            "MANIFEST.in",
            ".gitignore",
        )
    ]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }


def _read_lock(path: Path) -> dict:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock["protocol_version"] != PROTOCOL_VERSION
        or canonical_hash(lock["files"]) != lock["source_sha256"]
    ):
        raise ValueError("Invalid source lock")
    return lock


def active_source_lock() -> Path:
    pointer = LOCK.parent / "active-source.json"
    if not pointer.exists():
        return LOCK
    selection = json.loads(pointer.read_text(encoding="utf-8"))
    name = selection["lock_file"]
    if not re.fullmatch(r"source-lock-[a-z0-9][a-z0-9-]*\.json", name):
        raise ValueError("Invalid active source lock path")
    path = LOCK.parent / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != selection["lock_sha256"]:
        raise ValueError("Active source lock checksum mismatch")
    if _read_lock(path)["source_sha256"] != selection["source_sha256"]:
        raise ValueError("Active source revision mismatch")
    return path


def freeze_source(revision: str | None = None) -> Path:
    """Append an implementation revision; never replace an existing lock/archive."""
    if revision is None and LOCK.exists():
        raise FileExistsError("Original lock is immutable; use freeze-source --revision <name>")
    if revision is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", revision):
        raise ValueError("Source revision must use lowercase letters, digits, and hyphens")
    target = LOCK if revision is None else LOCK.parent / f"source-lock-{revision}.json"
    if target.exists():
        raise FileExistsError(f"Source revision already exists: {target}")
    files = source_files(ROOT)
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "files": files,
        "source_sha256": canonical_hash(files),
    }
    if revision is not None:
        original = _read_lock(LOCK)
        parent = active_source_lock()
        for filename in (
            "configs/full.toml",
            "configs/smoke.toml",
            "docs/VERIFICATION_PLAN.md",
            "constraints-gpu.txt",
        ):
            if files.get(filename) != original["files"].get(filename):
                raise ValueError(
                    f"Implementation-only amendment changed frozen choices: {filename}"
                )
        lock.update(
            {
                "source_revision": revision,
                "original_source_sha256": original["source_sha256"],
                "original_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                "parent_lock": parent.name,
                "parent_lock_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                "parent_source_sha256": _read_lock(parent)["source_sha256"],
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_bytes = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    with target.open("xb") as stream:
        stream.write(lock_bytes)
    prefix = PROTOCOL_VERSION if revision is None else f"{PROTOCOL_VERSION}-{revision}"
    archive = ROOT / "artifacts" / "source" / f"{prefix}-{lock['source_sha256']}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "lock_file": target.name,
        "source_sha256": lock["source_sha256"],
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }
    with ZipFile(archive, "x", ZIP_DEFLATED) as stream:
        for filename in files:
            stream.write(ROOT / filename, filename)
        stream.write(LOCK, LOCK.relative_to(ROOT).as_posix())
        if revision is not None:
            stream.write(target, target.relative_to(ROOT).as_posix())
            stream.writestr(
                (LOCK.parent / "active-source.json").relative_to(ROOT).as_posix(),
                json.dumps(selection, indent=2, sort_keys=True) + "\n",
            )
    if revision is not None:
        temporary = LOCK.parent / f".active-{uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, LOCK.parent / "active-source.json")
    return archive


def require_frozen_source() -> str:
    if not LOCK.exists():
        raise ValueError("Reviewed source is not frozen; run the freeze-source command")
    lock = _read_lock(active_source_lock())
    if "original_lock_sha256" in lock:
        if hashlib.sha256(LOCK.read_bytes()).hexdigest() != lock["original_lock_sha256"]:
            raise ValueError("Original source lock was modified")
        name = lock["parent_lock"]
        if name != LOCK.name and not re.fullmatch(r"source-lock-[a-z0-9][a-z0-9-]*\.json", name):
            raise ValueError("Invalid parent source lock path")
        parent = LOCK.parent / name
        if (
            hashlib.sha256(parent.read_bytes()).hexdigest() != lock["parent_lock_sha256"]
            or _read_lock(parent)["source_sha256"] != lock["parent_source_sha256"]
        ):
            raise ValueError("Parent source lock was modified")
    files = source_files(ROOT)
    if files != lock["files"] or canonical_hash(files) != lock["source_sha256"]:
        raise ValueError("Source/configuration changed since review; review and freeze again")
    return lock["source_sha256"]
