"""An implementation amendment must preserve the original immutable source freeze."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from watermark_robustness.protocol import active_source_lock, freeze_source, require_frozen_source


class SourceRevisionTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        for name, content in {
            "configs/full.toml": "full settings",
            "configs/smoke.toml": "smoke settings",
            "docs/VERIFICATION_PLAN.md": "frozen rules",
            "constraints-gpu.txt": "versions",
        }.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        lock = root / "protocols/tokenized-v2/source-lock.json"
        return root, lock

    def test_new_revision_preserves_original_and_becomes_training_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.fixture(directory)
            with (
                patch("watermark_robustness.protocol.ROOT", root),
                patch("watermark_robustness.protocol.LOCK", lock),
            ):
                original_archive = freeze_source()
                old_hash = require_frozen_source()
                old_lock_bytes, old_zip_bytes = lock.read_bytes(), original_archive.read_bytes()
                code = root / "src/watermark_robustness/verification.py"
                code.parent.mkdir(parents=True)
                code.write_text("# implementation addition")
                with self.assertRaisesRegex(ValueError, "changed since review"):
                    require_frozen_source()
                with self.assertRaises(FileExistsError):
                    freeze_source()
                archive = freeze_source("verification-v1")
                new_hash = require_frozen_source()
                self.assertNotEqual(new_hash, old_hash)
                self.assertEqual(active_source_lock().name, "source-lock-verification-v1.json")
                self.assertEqual(lock.read_bytes(), old_lock_bytes)
                self.assertEqual(original_archive.read_bytes(), old_zip_bytes)
                revised = json.loads(active_source_lock().read_text())
                self.assertEqual(revised["parent_source_sha256"], old_hash)
                self.assertEqual(revised["original_source_sha256"], old_hash)
                with ZipFile(archive) as stream:
                    for name, expected in revised["files"].items():
                        self.assertEqual(hashlib.sha256(stream.read(name)).hexdigest(), expected)
                    self.assertEqual(stream.read(lock.relative_to(root).as_posix()), old_lock_bytes)
                with self.assertRaises(FileExistsError):
                    freeze_source("verification-v1")
                self.assertEqual(require_frozen_source(), new_hash)
                pointer = lock.parent / "active-source.json"
                pointer.unlink()
                with self.assertRaisesRegex(ValueError, "changed since review"):
                    require_frozen_source()

    def test_amendment_rejects_methodology_changes_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.fixture(directory)
            with (
                patch("watermark_robustness.protocol.ROOT", root),
                patch("watermark_robustness.protocol.LOCK", lock),
            ):
                freeze_source()
                (root / "configs/full.toml").write_text("changed hyperparameters")
                with self.assertRaisesRegex(ValueError, "changed frozen choices"):
                    freeze_source("bad-amendment")
                self.assertFalse((lock.parent / "source-lock-bad-amendment.json").exists())
                with self.assertRaises(ValueError):
                    freeze_source("../escape")

    def test_tampered_active_pointer_and_original_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, lock = self.fixture(directory)
            with (
                patch("watermark_robustness.protocol.ROOT", root),
                patch("watermark_robustness.protocol.LOCK", lock),
            ):
                freeze_source()
                (root / "README.md").write_text("new implementation notes")
                freeze_source("verification-v1")
                pointer = lock.parent / "active-source.json"
                good = pointer.read_bytes()
                selection = json.loads(good)
                selection["lock_sha256"] = "0" * 64
                pointer.write_text(json.dumps(selection))
                with self.assertRaisesRegex(ValueError, "checksum"):
                    require_frozen_source()
                pointer.write_bytes(good)
                lock.write_bytes(lock.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "Original source lock was modified"):
                    require_frozen_source()


if __name__ == "__main__":
    unittest.main()
