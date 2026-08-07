"""Shared table-formatting utilities for paper-ready LaTeX output.

Design notes
~~~~~~~~~~~~
* The ``format_eval_table`` function is the main entry point for the two-metric
  evaluation table (MAE + F-beta).
* All per-cell significance formatting (stars + parenthesised statistic) lives
  in ``format_dm_cell`` so that both the per-comparison DM tables and the
  multi-seed dominance table look identical.
* ``to_latex_table`` is a thin wrapper around ``DataFrame.to_latex`` that
  enforces the project's booktabs options.
* Column formatters follow the ``save_VaR_backtesting_results`` pattern:
  ``df.to_latex(formatters={col: fn, ...})`` so that each column can be
  formatted differently while pandas still handles the table structure.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------

#: Display names for LaTeX table headers.
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "mae":       r"MAE",
    "mae_conditional": r"MAE | y > 0",
    "fbeta_2.0": r"F$_{\beta=2}$",
    "fbeta_2":   r"F$_{\beta=2}$",
    "r2_score":  r"$R^2$",
}

#: True when a higher value is *better* for the given metric.
METRIC_HIGHER_BETTER: dict[str, bool] = {
    "mae":       False,
    "mae_conditional": False,
    "fbeta_2.0": True,
    "fbeta_2":   True,
    "r2_score":  True,
}

#: Number of decimal places.
METRIC_PRECISION: dict[str, int] = {
    "mae":       1,
    "mae_conditional": 1,
    "r2_score": 2,
    "fbeta_2.0": 2,
    "fbeta_2":   2,
}

# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def escape_latex(s: str) -> str:
    """Escape the most common LaTeX special characters in a plain string."""
    return (
        s.replace("_", r"\_")
         .replace("%", r"\%")
         .replace("&", r"\&")
         .replace("#", r"\#")
    )


def significance_stars(p_val: float) -> str:
    """Return ``***``, ``**``, ``*``, or ``""`` based on *p_val*."""
    if p_val < 0.01:
        return "***"
    if p_val < 0.05:
        return "**"
    if p_val < 0.10:
        return "*"
    return ""


def format_dm_cell(
    statistic: float,
    p_val: float,
    value: float | None = None,
    *,
    percent: bool = False,
    value_decimals: int = 2,
    stat_decimals: int = 2,
) -> str:
    r"""Format a two-line DM / dominance cell matching the ``no_exog_comparison``
    table style::

        {value}{stars}
        ({statistic})

    where the second line is enclosed in parentheses and both lines share the
    same table cell (caller is responsible for the ``\\\\`` row separator).

    Parameters
    ----------
    statistic:
        DM test statistic (printed in parentheses on the second line).
    p_val:
        p-value used to determine significance stars.
    value:
        Optional effect-size value (e.g. percentage improvement or dominance
        share).  When *None*, only ``{stars}\\n({statistic})`` is returned.
    percent:
        When *True*, *value* is multiplied by 100 and a ``\%`` suffix is
        appended.
    value_decimals, stat_decimals:
        Decimal places for *value* and *statistic* respectively.
    """
    stars = significance_stars(p_val)
    if value is not None:
        if percent:
            val_str = f"{value * 100:.{value_decimals}f}\\%"
        else:
            val_str = f"{value:.{value_decimals}f}"
        first_line = f"{val_str}{stars}"
    else:
        first_line = stars

    second_line = f"({statistic:.{stat_decimals}f})"
    return f"{first_line} \\\\\n & & {second_line}"


def make_decimal_formatter(decimals: int = 2) -> Callable[[float], str]:
    """Return a formatter for plain decimal values with fixed precision."""

    def _fmt(v: float) -> str:
        if pd.isna(v):
            return ""
        return f"{v:.{decimals}f}"

    return _fmt


def make_percentage_formatter(decimals: int = 2, force_percentage: bool = False) -> Callable[[float], str]:
    """Return a formatter for percentage-like values.

    Values in ``[0, 1]`` are displayed as percentages. Other finite numeric
    values are displayed as plain decimals using the same precision. This keeps
    mixed tables readable while enforcing consistent decimal limits.
    """

    def _fmt(v: float) -> str:
        if pd.isna(v):
            return ""
        value = float(v)
        if 0.0 <= value <= 1.0 or force_percentage:
            return rf"{value * 100:.{decimals}f}\%"
        return f"{value:.{decimals}f}"

    return _fmt


# ---------------------------------------------------------------------------
# Best-value highlighting
# ---------------------------------------------------------------------------


def make_best_formatter(
    values: pd.Series,
    higher_is_better: bool,
    precision: int,
) -> Callable[[float], str]:
    """Return a per-column formatter that bolds the best cell.

    The returned callable is suitable for passing to
    ``DataFrame.to_latex(formatters={col: fn})``.

    Parameters
    ----------
    values:
        Float Series for the column (used to identify the best index).
    higher_is_better:
        ``True`` for metrics like F-beta; ``False`` for MAE.
    precision:
        Number of decimal places.
    """
    best_val: float = values.max() if higher_is_better else values.min()

    def _fmt(v: float) -> str:
        formatted = f"{v:.{precision}f}"
        if np.isclose(v, best_val):
            return rf"\textbf{{{formatted}}}"
        return formatted

    return _fmt


# ---------------------------------------------------------------------------
# Evaluation table (MAE + F-beta only)
# ---------------------------------------------------------------------------


def format_eval_table(
    eval_df: pd.DataFrame,
    metrics: tuple[str, ...] = ("mae", "fbeta_2.0"),
    model_display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pivot an evaluation DataFrame into a paper-ready table with two metrics.

    The input *eval_df* is the output of
    ``covariate_effect.evaluate.perform_tso_evaluations_rolling_window``:
    long-form with columns ``tso``, ``unique_id``, ``metric``, and one column
    per model.

    Returns a wide DataFrame with a three-level row index
    ``(TSO, Direction, Metric)`` and one column per model, with
    values already formatted as strings (best per-metric cell bolded).

    Parameters
    ----------
    eval_df:
        Long-form evaluation results.
    metrics:
        Subset of metric names to include (default: MAE and F-beta 2.0).
    model_display_names:
        Optional mapping ``{column_name: display_name}`` for renaming model
        columns in the output table.
    """
    meta = {"tso", "unique_id", "metric", "sparsity", "merge_key"}
    model_cols = [c for c in eval_df.columns if c not in meta]

    rows = []
    for (tso, uid, metric), grp in eval_df[
        eval_df["metric"].isin(metrics)
    ].groupby(["tso", "unique_id", "metric"]):
        if grp.empty:
            continue
        prec = METRIC_PRECISION.get(metric, 3)
        higher = METRIC_HIGHER_BETTER.get(metric, False)

        vals = {m: float(grp[m].iloc[0]) for m in model_cols if m in grp.columns}
        if not vals:
            continue
        best = max(vals, key=vals.get) if higher else min(vals, key=vals.get)

        row: dict[str, object] = {
            "TSO": tso,
            "Direction": uid,
            "Metric": METRIC_DISPLAY_NAMES.get(metric, metric),
        }
        for m, v in vals.items():
            display = m if model_display_names is None else model_display_names.get(m, m)
            cell = f"{v:.{prec}f}"
            if m == best:
                cell = rf"\textbf{{{cell}}}"
            row[display] = cell
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index(["TSO", "Direction", "Metric"])
    )


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def to_latex_table(
    df: pd.DataFrame,
    label: str = "",
    caption: str = "",
    formatters: dict[str, Callable] | None = None,
    column_format: str | None = None,
    final_form_callback: Callable[[str, pd.DataFrame], str] | None = None,
) -> str:
    """Render *df* to a full booktabs LaTeX tabular string.

    Parameters
    ----------
    df:
        DataFrame to render (string cells are not escaped again).
    label, caption:
        Optional LaTeX label / caption strings.
    formatters:
        Per-column formatter callables (same interface as
        ``DataFrame.to_latex(formatters=...)``).  Use :func:`make_best_formatter`
        to generate them automatically.
    column_format:
        Explicit LaTeX column spec (e.g. ``"lrrrr"``).  When *None*, pandas
        infers it automatically.
    """
    render_kwargs: dict = dict(
        escape=False,
        bold_rows=False,
        multirow=True,
        multicolumn=True,
        index_names=False,
        na_rep="",
    )
    if label:
        render_kwargs["label"] = label
    if caption:
        render_kwargs["caption"] = caption
    if formatters:
        render_kwargs["formatters"] = formatters
    if column_format:
        render_kwargs["column_format"] = column_format

    latex = df.to_latex(**render_kwargs)
    if final_form_callback is not None:
        latex = final_form_callback(latex, df)
    return latex


def add_index_names(latex: str, df: pd.DataFrame) -> str:
    """Add index names to a LaTeX table string."""
    if df.index.names:
        if df.columns.nlevels > 1:
            columns = df.columns.get_level_values(-1)
        else:
            columns = df.columns
        index_cols = len(df.index.names)
        existing_header_line = (
            r" {1,2}\& {1,2}".join([""] * index_cols) 
            + r" {1,2}\& {1,2}" 
            + r" {1,2}\& {1,2}".join(columns)
        )
        if re.search(existing_header_line, latex, flags=re.MULTILINE):
            index_names = df.index.names if df.index.names is not None else []
            new_header = " & ".join(index_names + list(columns))
            return re.sub(existing_header_line, new_header, latex, count=1, flags=re.MULTILINE)
        else:
            print(f"Warning: could not find expected header line in LaTeX output; index names not added. Expected pattern: {existing_header_line}")
            return latex
    return latex


def save_latex_table(
    df: pd.DataFrame,
    path: Path,
    **kwargs,
) -> None:
    """Write the LaTeX representation of *df* to *path*.

    All keyword arguments are forwarded to :func:`to_latex_table`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_latex_table(df, **kwargs))
