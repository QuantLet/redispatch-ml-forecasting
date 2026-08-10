#!/usr/bin/env python3
"""Download and verify the paper reproduction artifacts from Zenodo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

GROUPS = ("predictions", "interpretability", "ablations")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in (target, *target.parents):
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in artifact archives: {member.name}")
        archive.extractall(destination)


def _existing_members(archive_path: Path, destination: Path) -> list[Path]:
    with tarfile.open(archive_path, "r:gz") as archive:
        return [
            destination / member.name
            for member in archive.getmembers()
            if member.isfile() and (destination / member.name).exists()
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for group in GROUPS:
        parser.add_argument(f"--{group}", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("paper_artifacts.json"),
    )
    parser.add_argument("--download-dir", type=Path, default=Path(".paper_artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(GROUPS) if args.all else [group for group in GROUPS if getattr(args, group)]
    if not selected:
        raise SystemExit("Select at least one artifact group or pass --all.")

    root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    download_dir = args.download_dir if args.download_dir.is_absolute() else root / args.download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    for group in selected:
        record = manifest["artifacts"].get(group)
        if record is None:
            raise KeyError(f"Artifact group '{group}' is absent from {manifest_path}")
        url = str(record["url"])
        expected = str(record["sha256"])
        if url.startswith("REPLACE_") or expected.startswith("REPLACE_"):
            raise ValueError(
                f"The committed manifest has not been filled with the Zenodo {group} URL/checksum."
            )

        archive_path = download_dir / str(record["filename"])
        if archive_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {archive_path}. Pass --overwrite to continue."
            )
        if archive_path.exists():
            print(f"warning: replacing downloaded archive {archive_path}")

        with tempfile.NamedTemporaryFile(
            prefix=f".{archive_path.name}.", suffix=".tmp", dir=download_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            print(f"downloading {group}: {url}")
            try:
                with urlopen(url) as response:
                    shutil.copyfileobj(response, temporary)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise

        actual = sha256(temporary_path)
        if actual != expected:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {group}: expected {expected}, received {actual}"
            )
        existing_members = _existing_members(temporary_path, root)
        if existing_members and not args.overwrite:
            temporary_path.unlink(missing_ok=True)
            preview = ", ".join(str(path) for path in existing_members[:3])
            suffix = " ..." if len(existing_members) > 3 else ""
            raise FileExistsError(
                f"Refusing to overwrite {len(existing_members)} existing artifact files: "
                f"{preview}{suffix}. Pass --overwrite to continue."
            )
        if existing_members:
            print(f"warning: overwriting {len(existing_members)} existing artifact files")
        if archive_path.exists():
            archive_path.unlink()
        temporary_path.replace(archive_path)
        _safe_extract(archive_path, root)
        print(f"verified and extracted {archive_path.name}")


if __name__ == "__main__":
    main()
