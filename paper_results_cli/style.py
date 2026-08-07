"""Publication-ready matplotlib / seaborn styling and model colour registry.

Call :func:`apply_paper_style` once at programme start.  Then use
:func:`get_model_color` to look up colours in any plot, and call
:func:`save_color_registry` to write the palette to disk in both
machine-readable (JSON) and LaTeX-copyable formats.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .config import MODEL_DISPLAY_NAMES

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

#: Canonical hex colours keyed by *seed-agnostic* model names.
#: Any column name that contains a known base name (e.g. ``nhits_seed778``)
#: will resolve to the same colour via :func:`get_model_color`.
MODEL_COLORS: dict[str, str] = {
    "nhits":                        "#2196F3",   # blue
    "nbeatsx":                      "#4CAF50",   # green
    "tft":                          "#FF9800",   # orange
    "lstm":                         "#9C27B0",   # purple
    "auto_arima_rolling":           "#F44336",   # red
    "lightgbm_regression_rolling":  "#795548",   # brown
    "ridge_regression_rolling":     "#607D8B",   # blue-grey
    "naive_rolling":                "#9E9E9E",   # grey
    "moving_avg_hourly":            "#BDBDBD",   # light grey
    "null":                         "#E0E0E0",   # near-white (zero baseline)
}


def _base_name(col: str) -> str:
    """Strip a trailing ``_seed<digits>`` suffix.

    Examples
    --------
    >>> _base_name("nhits_seed778")
    'nhits'
    >>> _base_name("auto_arima_rolling")
    'auto_arima_rolling'
    """
    return re.sub(r"_seed\d+$", "", col)


def get_model_color(col: str) -> str:
    """Return the canonical hex colour for *col*, falling back to grey.

    Parameters
    ----------
    col:
        A model column name, possibly with a ``_seed<N>`` suffix.
    """
    return MODEL_COLORS.get(_base_name(col), "#BDBDBD")


def get_model_display_name(col: str) -> str:
    """Return publication display name for a model column.

    Seed suffixes are stripped before lookup, so ``nhits_seed778`` maps to
    ``NHiTS``. Unknown labels are returned unchanged.
    """
    base = _base_name(col)
    return MODEL_DISPLAY_NAMES.get(base, col)


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------


def apply_paper_style() -> None:
    """Configure matplotlib / seaborn for article-quality output.

    Settings applied
    ~~~~~~~~~~~~~~~~
    * Serif font, 9 pt body / 8 pt ticks
    * Only bottom and left spines
    * 300 DPI save, transparent background
    * Legend frame off (individual functions must still call
      :func:`no_legend` or ``ax.get_legend().remove()`` to suppress the
      legend *object* when one is generated)

    Also registers the ``"mae_diff"`` diverging colormap (red → white → green)
    used by the per-horizon MAE difference heatmap.
    """
    sns.set_style("ticks")
    mpl.rcParams.update(
        {
            # --- Typography ---
            "font.family":      "serif",
            "font.size":        9,
            "axes.labelsize":   9,
            "xtick.labelsize":  8,
            "ytick.labelsize":  8,
            # --- Spines ---
            "axes.spines.top":   False,
            "axes.spines.right": False,
            # --- Figures ---
            "figure.dpi":         150,
            "savefig.dpi":        300,
            "savefig.bbox":       "tight",
            "savefig.transparent": True,
            # --- Legend ---
            "legend.frameon": False,
        }
    )

    # Diverging colormap: red (negative diff) → white (zero) → green (positive)
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list(
            "mae_diff",
            ["#346f983d", "#f7f7f7", "#fdedbc55"],
        ),
        name="mae_diff",
        force=True,
    )


# ---------------------------------------------------------------------------
# Colour registry persistence
# ---------------------------------------------------------------------------


def save_color_registry(output_dir: Path) -> None:
    """Write the model colour palette to *output_dir* in two formats.

    Files created
    ~~~~~~~~~~~~~
    ``model_colors.json``
        Machine-readable mapping ``{model_name: "#rrggbb"}``.

    ``model_colors.tex``
        One ``\\definecolor`` command per model, ready to paste into a LaTeX
        preamble.  Requires ``\\usepackage[dvipsnames]{xcolor}``.

    Parameters
    ----------
    output_dir:
        Destination directory (created if it does not exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────────────────────
    (output_dir / "model_colors.json").write_text(
        json.dumps(MODEL_COLORS, indent=2) + "\n"
    )

    # ── LaTeX ────────────────────────────────────────────────────────────────
    lines: list[str] = [
        "% Model colour definitions - generated by paper_results_cli/style.py",
        "% Paste into LaTeX preamble after: \\usepackage[dvipsnames]{xcolor}",
        "",
    ]
    for model, hex_color in MODEL_COLORS.items():
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        # Build a valid LaTeX colour name: TitleCase, no underscores/hyphens.
        tex_name = (
            model.replace("_", " ")
            .replace("-", " ")
            .title()
            .replace(" ", "")
        )
        lines.append(
            f"\\definecolor{{{tex_name}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}"
            f"  % {model}: {hex_color}"
        )
    (output_dir / "model_colors.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def no_legend(ax: mpl.axes.Axes) -> None:
    """Remove the legend from *ax* if one exists."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def save_figure(
    fig: mpl.figure.Figure,
    path: Path,
    fmt: str = "png",
) -> None:
    """Save *fig* with paper-appropriate settings and close it.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    path:
        Destination path.  The suffix is **replaced** by *fmt*, so you may
        pass any base name.
    fmt:
        ``"png"`` (default) or ``"pdf"``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(f".{fmt}"),
        transparent=True,
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
