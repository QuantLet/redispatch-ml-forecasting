from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from paper_results_cli.data import (
    _read_benchmark_predictions_strict,
    read_per_window_predictions,
)


class PerWindowPredictionTests(unittest.TestCase):
    dataset = "paper_dataset"

    def _write_prediction(
        self,
        root: Path,
        *,
        best: bool,
        model: str = "nhits_seed778",
        window: int = 2,
        value: float = 1.25,
    ) -> None:
        directory_name = "predictions_best_checkpoint" if best else "predictions"
        directory = (
            root / "50hertz" / self.dataset / f"window_{window}"
            / "evaluation" / directory_name
        )
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "unique_id": ["up"],
            "ds": [pd.Timestamp("2025-03-01")],
            "horizon": [1],
            "y": [2.0],
            model: [value],
        }).to_parquet(directory / f"{directory_name}_{model}_window{window}.parquet")

    def test_reads_best_checkpoint_per_window_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_prediction(root, best=True)
            result = read_per_window_predictions(
                root, self.dataset, checkpoint_best=True, model_filter="seed778"
            )

        self.assertEqual(result.loc[0, "tso"], "50Hertz")
        self.assertEqual(result.loc[0, "window_index"], 2)
        self.assertEqual(result.loc[0, "nhits_seed778"], 1.25)

    def test_best_checkpoint_never_falls_back_to_last(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_prediction(root, best=False)
            with self.assertRaisesRegex(FileNotFoundError, "best-checkpoint"):
                read_per_window_predictions(
                    root, self.dataset, checkpoint_best=True, model_filter="seed778"
                )

    def test_conflicting_duplicate_predictions_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_prediction(root, best=True, model="nhits_seed778", value=1.0)
            self._write_prediction(root, best=True, model="alias_seed778", value=2.0)
            # Both files expose their own model columns, so this remains a valid wide row.
            result = read_per_window_predictions(
                root, self.dataset, checkpoint_best=True, model_filter="seed778"
            )
        self.assertEqual(set(result.columns) & {"nhits_seed778", "alias_seed778"}, {
            "nhits_seed778", "alias_seed778"
        })


class BenchmarkTests(unittest.TestCase):
    def test_ambiguous_benchmark_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "50hertz" / "paper_dataset" / "evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "benchmarks_a.csv").write_text("ds\n")
            (evaluation / "benchmarks_b.csv").write_text("ds\n")
            with self.assertRaisesRegex(FileNotFoundError, "exactly one benchmark"):
                _read_benchmark_predictions_strict(
                    root, "paper_dataset", pd.Timestamp("2025-03-01")
                )


if __name__ == "__main__":
    unittest.main()
