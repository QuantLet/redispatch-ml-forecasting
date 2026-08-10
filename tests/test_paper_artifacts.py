from __future__ import annotations

import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest


def _load_packager():
    path = Path(__file__).resolve().parents[1] / "scripts" / "package_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("package_paper_artifacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_paper_artifacts.py"
    spec = importlib.util.spec_from_file_location("download_paper_artifacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeterministicArchiveTests(unittest.TestCase):
    def test_same_inputs_produce_identical_archives(self) -> None:
        packager = _load_packager()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "nested" / "artifact.txt"
            source.parent.mkdir()
            source.write_text("deterministic paper artifact\n")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            packager.create_deterministic_archive(root, [source], first)
            source.touch()
            packager.create_deterministic_archive(root, [source], second)

            self.assertEqual(packager.sha256(first), packager.sha256(second))
            with tarfile.open(first, "r:gz") as archive:
                member = archive.getmember("nested/artifact.txt")
                self.assertEqual(member.mtime, 0)
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)

    def test_archive_entry_can_have_a_canonical_destination(self) -> None:
        packager = _load_packager()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "legacy-name.parquet"
            source.write_bytes(b"prediction")
            destination = root / "canonical.tar.gz"
            canonical = "outputs_paper/50hertz/dataset/window_2/prediction.parquet"
            packager.create_deterministic_archive(root, [(source, canonical)], destination)
            with tarfile.open(destination, "r:gz") as archive:
                self.assertEqual(archive.getnames(), [canonical])


class SafeExtractionTests(unittest.TestCase):
    def test_path_traversal_is_rejected(self) -> None:
        downloader = _load_downloader()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "unsafe.tar.gz"
            payload = root / "payload"
            payload.write_text("unsafe")
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(payload, arcname="../outside")
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                downloader._safe_extract(archive_path, root / "destination")


if __name__ == "__main__":
    unittest.main()
