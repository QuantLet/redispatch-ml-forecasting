#!/usr/bin/env python
"""
Explain --shift-hours semantics end-to-end for:
- training/runner.py (training-time slicing in model time)
- training/predict.py (prediction-time physical<->model time mapping)

This is a didactic script with a compact toy example. It mirrors the key logic
from prepare_shifted_dataset() and predict_with_shift_correction() but keeps the
numbers small so each mapping can be inspected by eye.

Run:
    python -m training.diagnose_shift_hours_runner_predict
"""

from __future__ import annotations

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def build_toy_data(n: int = 20) -> pd.DataFrame:
    """Build a tiny single-series dataset in physical time."""
    ds = pd.date_range("2024-01-01 00:00", periods=n, freq="h")
    return pd.DataFrame(
        {
            "unique_id": "up",
            "ds": ds,
            "y": [100 + i for i in range(n)],
            # Future exogenous: known for target timestamps (e.g., forecast).
            "wind_forecast": [1000 + i for i in range(n)],
            # Historical exogenous: only known when that physical time happens.
            "redispatch_actual_lagproxy": [2000 + i for i in range(n)],
        }
    )


def prepare_shifted_like_data_prep(df: pd.DataFrame, shift_hours: int) -> pd.DataFrame:
    """
    Mimic prepare_shifted_dataset() for this toy schema:
    - shift y and future covariates by -S
    - keep historical covariates unshifted
    """
    out = df.sort_values(["unique_id", "ds"]).copy()
    g = out.groupby("unique_id", sort=False)

    out["y"] = g["y"].shift(-shift_hours)
    out["wind_forecast"] = g["wind_forecast"].shift(-shift_hours)

    out = out.dropna(subset=["y", "wind_forecast"]).reset_index(drop=True)
    return out


def print_core_mapping(df_unshifted: pd.DataFrame, df_shifted: pd.DataFrame, s: int) -> None:
    print("=" * 92)
    print("A) ROW-LEVEL SEMANTICS AFTER prepare_shifted_dataset(..., shift_hours=S)")
    print("=" * 92)
    print("For each model-time row with index t (timestamp ds=t):")
    print("  y_row(t)         = y_phys(t + S)")
    print("  future_row(t)    = x_futr_phys(t + S)")
    print("  hist_row(t)      = x_hist_phys(t)")
    print()

    view = df_shifted[["ds", "y", "wind_forecast", "redispatch_actual_lagproxy"]].head(8).copy()
    view["physical_target_time"] = view["ds"] + pd.Timedelta(hours=s)
    print("First 8 shifted rows:")
    print(view.to_string(index=True))
    print()

    i = 2
    row = df_shifted.iloc[i]
    phys_t = row["ds"] + pd.Timedelta(hours=s)
    unshift_match = df_unshifted[df_unshifted["ds"] == phys_t].iloc[0]
    hist_match = df_unshifted[df_unshifted["ds"] == row["ds"]].iloc[0]
    print(f"Concrete check for shifted row i={i}:")
    print(f"  model ds                : {row['ds']}")
    print(f"  physical target time    : {phys_t} = ds + S")
    print(f"  y at shifted row        : {row['y']}  (matches physical y {unshift_match['y']})")
    print(
        f"  future cov at shifted row: {row['wind_forecast']}  "
        f"(matches physical future cov {unshift_match['wind_forecast']})"
    )
    print(
        f"  historical cov at shifted row: {row['redispatch_actual_lagproxy']}  "
        f"(matches physical hist cov at ds {hist_match['redispatch_actual_lagproxy']})"
    )


def print_why_hist_not_shifted(s: int) -> None:
    print()
    print("=" * 92)
    print("B) WHY historical exogenous ARE NOT target-shifted")
    print("=" * 92)
    print("Historical exogenous are kept on model-time ds because they are context available")
    print("at prediction decision time, not known for future physical timestamps.")
    print()
    print("If they were shifted by S as well, row ds=t would use x_hist_phys(t+S),")
    print("which leaks information that is unavailable at prediction decision time.")
    print()
    print("Interpretation with lead time S:")
    print("  decision/model time t      -> row key ds=t")
    print("  first predicted physical y -> y_phys(t+S)")
    print("  historical context allowed -> up to physical t")
    print("  future covariates allowed  -> forecasts known for physical t+S...t+S+h-1")


def show_runner_time_slicing(s: int) -> None:
    print()
    print("=" * 92)
    print("C) runner.py split conversion: physical boundaries -> model-time boundaries")
    print("=" * 92)

    train_start = pd.Timestamp("2024-01-01 06:00")
    valid_start = pd.Timestamp("2024-01-01 12:00")
    test_start = pd.Timestamp("2024-01-01 16:00")

    print("Configured boundaries in physical time:")
    print(f"  train_start={train_start}, valid_start={valid_start}, test_start={test_start}")
    print("runner.py subtracts S before slicing shifted df:")
    print(f"  train_start_ts = train_start - S = {train_start - pd.Timedelta(hours=s)}")
    print(f"  valid_start_ts = valid_start - S = {valid_start - pd.Timedelta(hours=s)}")
    print(f"  test_start_ts  = test_start  - S = {test_start - pd.Timedelta(hours=s)}")
    print()
    print("Reason:")
    print("  shifted rows are indexed by model time ds=t, but represent physical target t+S.")
    print("  So physical [train_start, test_start) corresponds to model-time")
    print("  [train_start-S, test_start-S).")


def simulate_one_prediction_stride(
    df_shifted: pd.DataFrame,
    df_unshifted: pd.DataFrame,
    shift_hours: int,
    forecast_horizon: int,
    pred_start_physical: pd.Timestamp,
) -> None:
    """
    Mimic one stride from predict_with_shift_correction() and print each map.

    We do not call NeuralForecast here; we only trace which rows are selected and
    how timestamps are relabeled before/after prediction.
    """
    print()
    print("=" * 92)
    print("D) predict.py one-stride mapping (physical <-> model time)")
    print("=" * 92)

    model_pred_date = pred_start_physical - pd.Timedelta(hours=shift_hours)
    phys_start = model_pred_date + pd.Timedelta(hours=shift_hours)
    phys_end = phys_start + pd.Timedelta(hours=forecast_horizon)

    print(f"Requested first physical prediction timestamp pred_start={pred_start_physical}")
    print(f"predict.py sets model_pred_date = pred_start - S = {model_pred_date}")
    print()

    hist_df = df_shifted[df_shifted["ds"] < model_pred_date].copy()
    futr_phys = df_unshifted.loc[
        (df_unshifted["ds"] >= phys_start) & (df_unshifted["ds"] < phys_end),
        ["unique_id", "ds", "wind_forecast"],
    ].copy()

    print("1) Historical context from shifted df in model time")
    print(f"   hist condition: ds < {model_pred_date}")
    print("   last 5 historical rows:")
    print(hist_df[["ds", "y", "redispatch_actual_lagproxy"]].tail(5).to_string(index=False))
    print()

    print("2) Future covariates taken at physical timestamps")
    print(f"   physical window: [{phys_start}, {phys_end})")
    print(futr_phys.to_string(index=False))
    print()

    futr_model = futr_phys.copy()
    futr_model["ds"] = futr_model["ds"] - pd.Timedelta(hours=shift_hours)

    print("3) Future covariates relabeled to model-time scaffold (ds -= S)")
    print(futr_model.to_string(index=False))
    print()

    print("4) After nf.predict(), predict.py shifts prediction ds back to physical (ds += S)")
    print("   So output timestamps are again in physical time and directly comparable to actual y.")


def print_training_mapping_statement(s: int, h: int) -> None:
    print()
    print("=" * 92)
    print("E) What function is learned? (conceptual mapping)")
    print("=" * 92)
    print("At model decision time t, the model receives:")
    print("  - target/history context ending before t (already shifted y series in model time)")
    print("  - historical exogenous on model time (physical t and earlier)")
    print("  - future exogenous aligned to physical target times t+S ... t+S+h-1")
    print()
    print("For row semantics, your equation is correct:")
    print("  x_hist(t) together with x_futr(t+S) helps predict y(t+S).")
    print()
    print("For the multi-step Nixtla window, a concise view is:")
    print(f"  F( history<=t, futr[t..t+{h-1}] in model-time labels ) -> y[t..t+{h-1}] (model-time labels)")
    print("  and then labels map back to physical time by adding S.")
    print()
    print("Equivalent physical-time interpretation:")
    print(f"  decision at time t predicts physical targets in [t+{s}, t+{s+h-1}].")


def main() -> None:
    s = 3
    h = 4

    print("\n" + "#" * 92)
    print("DETAILED WALKTHROUGH OF --shift-hours IN runner.py AND predict.py")
    print("#" * 92)
    print(f"Toy setup: shift_hours S={s}, forecast_horizon h={h}\n")

    df_unshifted = build_toy_data(n=20)
    df_shifted = prepare_shifted_like_data_prep(df_unshifted, shift_hours=s)

    print_core_mapping(df_unshifted, df_shifted, s=s)
    print_why_hist_not_shifted(s=s)
    show_runner_time_slicing(s=s)

    pred_start_physical = pd.Timestamp("2024-01-01 16:00")
    simulate_one_prediction_stride(
        df_shifted=df_shifted,
        df_unshifted=df_unshifted,
        shift_hours=s,
        forecast_horizon=h,
        pred_start_physical=pred_start_physical,
    )

    print_training_mapping_statement(s=s, h=h)

    print()
    print("=" * 92)
    print("F) Short answer to your 4 questions")
    print("=" * 92)
    print("1) Historical exogenous are not target-shifted to avoid leakage.")
    print("2) Yes: row-wise semantics are x_hist_t + x_futr_{t+S} -> y_{t+S}.")
    print("3) Physical time = when quantity exists; model/prediction time = ds index after shift.")
    print("   They are related by physical_time = model_time + S.")
    print("4) Prediction is exactly: physical->model shift for stride start, build history in model time,")
    print("   fetch future covariates in physical time, relabel to model time, predict, then shift output")
    print("   timestamps back to physical time.")


if __name__ == "__main__":
    main()
