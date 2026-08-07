#!/usr/bin/env python
"""
Diagnostic script: trace what circular_shift + target_shift does to a small
series and show where future information leaks into the validation split.

Run:  python -m training.diagnose_circular_shift
"""

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 200)


def main():
    # ── Parameters ────────────────────────────────────────────────────────────
    N          = 24          # total hours in this toy series
    T_h        = 16          # "training-set length" used as circular period
    K_days     = 1           # circular shift = 1 day = K_h = 24 hours → K_eff = 24 % 16 = 8
    K_h        = K_days * 24 # = 24
    S          = 6           # target shift hours (--shift-hours)
    val_hours  = 4           # last 4 hours of the window are validation
    train_hours = N - val_hours  # = 20 hours in the window (train + val)

    # Physical boundaries inside the window:
    # training: t = 0..15  (T_h = 16)
    # validation: t = 16..19
    # (test is outside the window; not shown here)

    print("=" * 90)
    print("PARAMETERS")
    print(f"  Series length N        = {N}")
    print(f"  Circular period T_h    = {T_h}")
    print(f"  K_days = {K_days}   →   K_h = {K_h}   →   K_eff = {K_h % T_h}")
    print(f"  Target shift S         = {S}")
    print(f"  Train+val window       = {train_hours} hours  (train {train_hours - val_hours} + val {val_hours})")
    print("=" * 90)

    K_eff = K_h % T_h  # = 8

    # ── Build a toy series ────────────────────────────────────────────────────
    t = np.arange(N)
    y_original       = 100.0 + t.astype(float)   # target: 100, 101, ..., 123
    forecast_cov     = 1000.0 + t.astype(float)   # future covariate (e.g. wind_forecast)
    historical_cov   = 2000.0 + t.astype(float)   # historical covariate (e.g. bloomberg_*)
    ds = pd.date_range("2024-01-01", periods=N, freq="h")

    df_orig = pd.DataFrame({
        "ds": ds, "y": y_original, "unique_id": "up",
        "wind_forecast": forecast_cov,       # future cov (contains "forecast")
        "bloomberg_idx": historical_cov,     # historical cov
    })

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Circular shift (applied BEFORE prepare_shifted_dataset)
    # ══════════════════════════════════════════════════════════════════════════
    # _shift_array formula:  indices = (arange(N) - K_eff) % T_eff
    #                        T_eff = min(T_h, N) = min(16, 24) = 16
    T_eff = min(T_h, N)
    indices = (np.arange(N) - K_eff) % T_eff
    # Both future AND historical covariates are shifted (only y, ds, uid, calendar excluded)
    circ_forecast  = forecast_cov[indices]
    circ_bloomberg = historical_cov[indices]

    df_circ = df_orig.copy()
    df_circ["wind_forecast"] = circ_forecast
    df_circ["bloomberg_idx"] = circ_bloomberg

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Target shift (prepare_shifted_dataset with shift_hours = S)
    # ══════════════════════════════════════════════════════════════════════════
    # y and FUTURE covariates shifted by  .shift(-S)  per group
    # historical covariates are NOT target-shifted
    # Calendar features recomputed on ds + S  (we ignore calendar here)

    df_shifted = df_circ.copy()
    df_shifted["y"]             = df_shifted["y"].shift(-S)
    df_shifted["wind_forecast"] = df_shifted["wind_forecast"].shift(-S)
    # bloomberg_idx is historical → NOT target-shifted
    df_shifted = df_shifted.dropna(subset=["y"])  # drops last S rows

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2b: Prediction df (target shift only, NO circular shift)
    # ══════════════════════════════════════════════════════════════════════════
    df_pred = df_orig.copy()
    df_pred["y"]             = df_pred["y"].shift(-S)
    df_pred["wind_forecast"] = df_pred["wind_forecast"].shift(-S)
    df_pred = df_pred.dropna(subset=["y"])

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE: Normal pipeline (no circular shift, just target shift)
    # ══════════════════════════════════════════════════════════════════════════
    df_baseline = df_orig.copy()
    df_baseline["y"]             = df_baseline["y"].shift(-S)
    df_baseline["wind_forecast"] = df_baseline["wind_forecast"].shift(-S)
    df_baseline = df_baseline.dropna(subset=["y"])

    # ══════════════════════════════════════════════════════════════════════════
    #  PRINT COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    effective_N = len(df_shifted)   # = N - S
    window_end = train_hours  # = 20 (but capped by effective_N = 18)
    window_end = min(window_end, effective_N)

    # For the window [0, window_end) the last val_hours are validation
    train_end = window_end - val_hours

    print()
    print("─" * 90)
    print("COLUMN LEGEND:")
    print("  t      = positional index in the series (= hours since start)")
    print("  ds     = timestamp")
    print("  y_phys = physical target hour (ds + S) - what the model is actually predicting")
    print("  y      = target value after target-shift")
    print("  wf     = wind_forecast (future covariate) after shifts")
    print("  bb     = bloomberg_idx (historical covariate) after shifts")
    print("  circ_i = index from which the circular shift reads this row's covariates")
    print("  split  = TRAIN / VAL / (beyond window)")
    print("─" * 90)
    print()

    print("══════════════════════════════════════════════════════════════════")
    print("BASELINE: target shift only (what train_pipeline.py does)")
    print("══════════════════════════════════════════════════════════════════")
    _print_table(df_baseline, S, indices=None, train_end=train_end, window_end=window_end)

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("CIRCULAR + TARGET SHIFT (what circular_shift_training.py does)")
    print("══════════════════════════════════════════════════════════════════")
    _print_table(df_shifted, S, indices=indices, train_end=train_end, window_end=window_end)

    # ══════════════════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 90)
    print("ANALYSIS OF DATA LEAKAGE / ALIGNMENT ISSUES")
    print("=" * 90)

    # Issue 1: Circular shift wraps val/test positions back into training period
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│  ISSUE 1: VALIDATION COVARIATES WRAP INTO TRAINING PERIOD              │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print()
    print(f"  T_h (circular period) = {T_h},  T_eff = {T_eff}")
    print(f"  Training positions:   t = 0 .. {train_end - 1}")
    print(f"  Validation positions: t = {train_end} .. {window_end - 1}")
    print()
    print("  The circular index formula is: src = (t - K_eff) % T_eff")
    print()
    for vt in range(train_end, window_end):
        src = (vt - K_eff) % T_eff
        region = "TRAINING" if src < train_end else "VALIDATION"
        print(f"    val position t={vt}: reads covariates from src={src}  ({region} period)")
    print()
    print("  → Validation rows get TRAINING-PERIOD covariate values.")
    print("    This is NOT data leakage (no future info leaks backward),")
    print("    but it makes the validation loss UNRELIABLE because the")
    print("    model validates on (real y) paired with (wrong covariates).")
    print("    Early stopping / model selection is therefore compromised.")

    # Issue 2: Historical covariates are also circularly shifted
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│  ISSUE 2: HISTORICAL COVARIATES ARE CIRCULARLY SHIFTED BUT NOT         │")
    print("│           TARGET-SHIFTED - DOUBLE MISALIGNMENT                         │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print()
    print("  classify_covariates() splits columns into:")
    print("    future = columns with 'forecast' or starting with 'day_ahead_price'")
    print("    historical = everything else (bloomberg, lag, rolling mean, etc.)")
    print()
    print("  prepare_shifted_dataset only target-shifts y + FUTURE covariates.")
    print("  Historical covariates are left at their original position.")
    print()
    print("  In the BASELINE pipeline:")
    print("    At row ds=t, historical covs are from physical time t → aligned with ds.")
    print()
    print("  In the CIRCULAR pipeline:")
    print("    Historical covs were circularly shifted in step 1, so at row ds=t,")
    print("    hist_cov comes from position (t - K_eff) % T_eff.")
    print("    But prepare_shifted_dataset does NOT target-shift them, so their")
    print("    temporal reference is still 'ds=t' (model time) not 'ds+S' (physical).")
    print()
    print("    This is the same as the baseline - hist covs are always aligned")
    print("    with model-time ds, not physical target time ds+S. The circular")
    print("    shift just permutes WHICH values appear at each position.")
    print("    → No extra misalignment beyond the intended augmentation.")

    # Issue 3: Can double shifting help?
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│  ISSUE 3: WHY COULD THE CIRCULAR-SHIFT MODEL APPEAR BETTER?           │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print()
    print("  The most likely explanation is NOT data leakage but a side-effect")
    print("  of the corrupted validation split:")
    print()
    print("  1. Circular shift gives validation rows WRONG covariates (from training).")
    print("  2. The model's validation loss becomes artificially HIGHER (noisy).")
    print("  3. Early stopping therefore runs LONGER (patience not triggered).")
    print("  4. The model trains for more steps → potentially better generalisation")
    print("     on the REAL test set (where pred_df uses correct covariates).")
    print()
    print("  This is an unfair advantage over the baseline, which stops early")
    print("  using a correctly-measured validation loss.")
    print()
    print("  Alternatively, if the model is already overfitting on the baseline,")
    print("  the corrupted validation acts like implicit regularisation.")
    print()
    print("  TO VERIFY: compare the number of training steps between circular-shift")
    print("  and baseline runs in W&B (trainer/global_step at run end).")

    # Proposed fix
    print()
    print("┌─────────────────────────────────────────────────────────────────────────┐")
    print("│  PROPOSED FIX                                                          │")
    print("└─────────────────────────────────────────────────────────────────────────┘")
    print()
    print("  Only circularly shift the TRAINING portion of each window - leave the")
    print("  validation slice with real covariates.  This requires a per-window")
    print("  transform (the runner.py per_window_transform callback):")
    print()
    print("    def per_window_circ_shift(window_df, train_hours):")
    print("        train_mask = window_df.index[:train_hours_per_uid]  # per group")
    print("        window_df.loc[train_mask, shift_cols] = circular_shifted_values")
    print("        # validation portion untouched")
    print("        return window_df")
    print()
    print("  With this fix, validation uses real covariates and early stopping")
    print("  works correctly.  The circular augmentation is still applied to")
    print("  the training data as intended.")
    print()


def _print_table(df, S, indices, train_end, window_end):
    header = f"{'t':>3} {'ds':>20} {'y_phys':>8} {'y':>8} {'wf':>8} {'bb':>8}"
    if indices is not None:
        header += f"  {'circ_i':>6}"
    header += f"  {'split':>8}"
    print(header)
    print("─" * len(header))
    for i, (_, row) in enumerate(df.iterrows()):
        y_phys = f"t+{S}={i+S}"
        split = "TRAIN" if i < train_end else ("VAL" if i < window_end else "")
        line = (
            f"{i:>3} {str(row['ds']):>20} {y_phys:>8} {row['y']:>8.0f} "
            f"{row['wind_forecast']:>8.0f} {row['bloomberg_idx']:>8.0f}"
        )
        if indices is not None:
            src = indices[i] if i < len(indices) else -1
            line += f"  {src:>6}"
        line += f"  {split:>8}"
        print(line)


if __name__ == "__main__":
    main()
