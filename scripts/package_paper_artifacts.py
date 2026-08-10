#!/usr/bin/env python3
"""Build deterministic archives for the paper's Zenodo reproduction record."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

GROUPS = ("predictions", "interpretability", "ablations")
TSOS = ("50hertz", "amprion", "tennet_de", "transnetbw")
WINDOWS = tuple(range(2, 14))
DATASET = "basic_day_ahead_price_wind_pv_production_consumption_sce"
TSO_DISPLAY = {
    "50hertz": "50Hertz",
    "amprion": "Amprion",
    "tennet_de": "TenneT_DE",
    "transnetbw": "TransnetBW",
}
ArchiveEntry = tuple[Path, str]

PREDICTION_SPECS = (
    ("outputs_paper", DATASET, 778, ("lstm", "nbeatsx", "nhits", "tft"), "standard"),
    ("outputs_shifted_targets_17_paper", DATASET, 778, ("nhits",), "ig"),
    ("outputs_no_covariates", "basic", 778, ("nhits",), "ig"),
    ("outputs_nhits_only_seed860_paper", DATASET, 860, ("nhits",), "ig"),
    ("outputs_shifted_targets_17_seed860_paper", DATASET, 860, ("nhits",), "ig"),
    ("outputs_no_covariates_nhits_only_seed860", "basic", 860, ("nhits",), "ig"),
)


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required paper artifact is missing: {path}")
    return path


def prediction_files(root: Path) -> list[ArchiveEntry]:
    files: list[ArchiveEntry] = []
    for output_name, dataset, seed, models, source_kind in PREDICTION_SPECS:
        for tso in TSOS:
            for window in WINDOWS:
                for model in models:
                    dataset_dir = root / output_name / tso / dataset
                    if source_kind == "standard":
                        matches = sorted((dataset_dir / "evaluation").glob(
                            f"predictions_{model}_seed{seed}_{TSO_DISPLAY[tso]}_"
                            f"*_window{window}_best_checkpoint.parquet"
                        ))
                    else:
                        matches = sorted((
                            dataset_dir / f"window_{window}" / "evaluation"
                            / "ig_preds_best_checkpoint"
                        ).glob(
                            f"ig_preds_best_checkpoint_{model}_seed{seed}_window{window}.parquet"
                        ))
                    if len(matches) != 1:
                        raise FileNotFoundError(
                            f"Expected one best-checkpoint prediction for {output_name}/"
                            f"{tso}/window_{window}/{model}_seed{seed}; found {len(matches)}."
                        )
                    archive_name = (
                        f"{output_name}/{tso}/{dataset}/window_{window}/evaluation/"
                        f"predictions_best_checkpoint/predictions_best_checkpoint_"
                        f"{model}_seed{seed}_window{window}.parquet"
                    )
                    files.append((matches[0], archive_name))

    for tso in TSOS:
        evaluation_dir = root / "outputs_paper" / tso / DATASET / "evaluation"
        matches = sorted(evaluation_dir.glob("benchmarks_*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one benchmark CSV in {evaluation_dir}; "
                f"found {len(matches)}."
            )
        files.append((matches[0], matches[0].relative_to(root).as_posix()))
    return files


def interpretability_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for tso in TSOS:
        for window in WINDOWS:
            model_dir = (
                root / "outputs_paper" / tso / DATASET / f"window_{window}"
                / "evaluation" / "ig_raw_best_checkpoint"
                / f"nhits_seed778_window{window}"
            )
            model_files = sorted(path for path in model_dir.glob("stride_*") if path.is_file())
            if not model_files:
                raise FileNotFoundError(f"No NHiTS IG tensors found in {model_dir}")
            files.extend(model_files)
    return files


def ablation_files(root: Path) -> list[Path]:
    files: list[Path] = []
    input_root = root / "input_size_ablation_paper"
    for tso_key, tso_name in (
        ("50hertz", "50Hertz"),
        ("amprion", "Amprion"),
        ("tennet_de", "TenneT_DE"),
        ("transnetbw", "TransnetBW"),
    ):
        matches = sorted((input_root / tso_key).glob(
            f"validation_predictions_{DATASET}_{tso_name}_k2_checkpoint_best.csv"
        ))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one input-size validation CSV for {tso_name}; found {len(matches)}."
            )
        files.append(matches[0])

    benchmark_root = root / "benchmark_ablation_paper"
    for tso_name in ("50Hertz", "Amprion", "TenneT_DE", "TransnetBW"):
        tso_dir = benchmark_root / tso_name
        files.append(_require_file(tso_dir / f"benchmark_ablation_{DATASET}_{tso_name}_k2.csv"))
        files.append(_require_file(tso_dir / f"best_params_{DATASET}_{tso_name}.json"))
    return files


SELECTORS = {
    "predictions": prediction_files,
    "interpretability": interpretability_files,
    "ablations": ablation_files,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(entry: Path | ArchiveEntry) -> Path:
    return entry[0] if isinstance(entry, tuple) else entry


def _arcname(root: Path, entry: Path | ArchiveEntry) -> str:
    return entry[1] if isinstance(entry, tuple) else entry.relative_to(root).as_posix()


def create_deterministic_archive(
    root: Path,
    files: list[Path] | list[ArchiveEntry],
    destination: Path,
) -> None:
    """Create a byte-reproducible gzip-compressed tar archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for entry in sorted(files, key=lambda item: _arcname(root, item)):
                        path = _source(entry)
                        relative = _arcname(root, entry)
                        info = archive.gettarinfo(str(path), arcname=relative)
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mode = 0o644
                        info.pax_headers = {}
                        with path.open("rb") as source:
                            archive.addfile(info, source)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("zenodo_upload"))
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, object]] = {}
    sums: list[str] = []
    for group in args.groups:
        destination = output_dir / f"paper_{group}.tar.gz"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite.")
        files = SELECTORS[group](root)
        total = sum(_source(entry).stat().st_size for entry in files)
        print(f"{group}: {len(files)} files, {total / 1024 / 1024:.2f} MiB uncompressed")
        create_deterministic_archive(root, files, destination)
        checksum = sha256(destination)
        print(f"created {destination} ({checksum})")
        sums.append(f"{checksum}  {destination.name}")
        records[group] = {
            "filename": destination.name,
            "sha256": checksum,
            "url": "REPLACE_WITH_ZENODO_DIRECT_URL",
            "top_level_paths": sorted({_arcname(root, entry).split("/", 1)[0] for entry in files}),
        }

    (output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    manifest = {
        "schema_version": 1,
        "doi": "REPLACE_WITH_VERSION_SPECIFIC_ZENODO_DOI",
        "artifacts": records,
    }
    (output_dir / "paper_artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {output_dir / 'SHA256SUMS'}")
    print(f"wrote {output_dir / 'paper_artifacts.json'}")


if __name__ == "__main__":
    main()
