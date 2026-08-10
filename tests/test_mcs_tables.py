from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from paper_results_cli.config import PaperConfig
from paper_results_cli.modules import mcs


class McsTableTests(unittest.TestCase):
    def test_fbeta_is_computed_separately_for_dense_and_sparse_series(self) -> None:
        predictions = pd.DataFrame(
            {
                "tso": ["A"] * 8,
                "unique_id": ["dense"] * 4 + ["sparse"] * 4,
                "y": [1, 0, 2, 1, 0, 0, 0, 1],
            }
        )
        config = PaperConfig(models=["nhits"], dm_sparsity_threshold=0.7)

        def fake_dominance(subset, _config, _models):
            series = subset["unique_id"].unique().item()
            return pd.Series({"NHiTS": 0.25 if series == "dense" else 0.75})

        with patch.object(mcs, "_fbeta_dominance", side_effect=fake_dominance):
            result = mcs._fbeta_dominance_by_sparsity(
                predictions, config, ["nhits"]
            )

        self.assertEqual(result.loc["NHiTS", "Dense"], 0.25)
        self.assertEqual(result.loc["NHiTS", "Sparse"], 0.75)

    def test_series_table_uses_x_only_for_included_status(self) -> None:
        predictions = pd.DataFrame(
            {
                "tso": ["A", "A"],
                "unique_id": ["up", "down"],
                "ds": pd.to_datetime(["2025-03-01", "2025-03-01"]),
                "horizon": [1, 1],
                "y": [1.0, 0.0],
                "nhits_seed778": [1.0, 0.0],
            }
        )
        statuses = pd.DataFrame(
            {
                "models": ["nhits_seed778", "nhits_seed778"],
                "status": ["included", "excluded"],
                "tso": ["A", "A"],
                "direction": ["up", "down"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            config = PaperConfig(
                models=["nhits"], output_dir=Path(temporary)
            )
            config.make_output_dirs()
            with patch.object(mcs, "compare_models_mcs", return_value=statuses):
                result = mcs.run_mcs_inclusion_by_series(predictions, config)

            latex = (config.tables_dir / "mcs_inclusion_by_series.tex").read_text()
            self.assertTrue(result.loc["NHiTS", ("A", "up")])
            self.assertFalse(result.loc["NHiTS", ("A", "down")])
            self.assertEqual(latex.count(r"\textbf{X}"), 2)  # caption and included cell


if __name__ == "__main__":
    unittest.main()
