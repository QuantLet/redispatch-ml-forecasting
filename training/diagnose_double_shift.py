#!/usr/bin/env python
"""
Diagnostic: combined target-shift + circular-shift on past & future covariates.

This script builds a small, self-contained example that traces every row
through *both* transformations and shows:

    1. What the raw data looks like.
    2. After the **target shift** (prepare_shifted_dataset logic):
       y and future covariates move by -S positions; historical covariates
       stay in place; calendar is recomputed on physical target time.
    3. After the **circular shift** (per-window callback):
       all non-calendar / non-meta / non-target columns are circularly
       permuted by K_h positions within the training portion.
    4. Why the two shifts do NOT cancel each other: the target shift changes
       *which* y the model predicts, while the circular shift scrambles
       *which* covariate vector accompanies that y.

Run:
    python -m training.diagnose_double_shift
"""

from __future__ import annotations

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
N_HOURS = 24          # total series length (1 day)
S       = 6           # target shift (hours)
K_DAYS  = 1           # circular shift (days)
K_H     = K_DAYS * 24 # circular shift in hours
TRAIN_H = 13          # first 13 hours are "training"
# After target shift, the last S rows per group become NaN and are dropped,
# leaving N_HOURS - S = 18 usable rows.  Of those, the first TRAIN_H = 13
# are training and the remaining 5 are validation.

SEPARATOR = "=" * 90

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build raw data
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("1. RAW DATA  (before any shifting)")
print(SEPARATOR)

ds = pd.date_range("2024-06-01", periods=N_HOURS, freq="h")
raw = pd.DataFrame({
    "ds":         ds,
    "unique_id":  "TSO_up",
    "y":          np.arange(N_HOURS) * 10,              # 0, 10, 20, ..., 230
    "wind_fcst":  1000 + np.arange(N_HOURS),            # future cov (contains "forecast")
    "price_da":   2000 + np.arange(N_HOURS),             # future cov (NOT shifted in our example – see note below)
    "wind_lag1":  3000 + np.arange(N_HOURS),            # historical cov (lag)
    "bloomberg":  4000 + np.arange(N_HOURS),            # historical cov
})

# For this toy example we treat:
#   wind_fcst    → future covariate  (contains "forecast")
#   price_da     → historical covariate (name doesn't start with "day_ahead_price")
#   wind_lag1    → historical covariate
#   bloomberg    → historical covariate
#
# In the real pipeline, classify_covariates checks:
#   "forecast" in name  → future
#   name.startswith("day_ahead_price") → future
#   else → historical

print(raw.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Target shift  (prepare_shifted_dataset logic)
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print(f"2. AFTER TARGET SHIFT  (S = {S} hours)")
print("   y and future covariates shifted by -.shift({S})  →  each row's y")
print("   now represents the physical value S hours later.")
print("   Historical covariates are untouched.")
print("   Calendar would be recomputed on ds + S (omitted here for clarity).")
print(SEPARATOR)

target_shifted = raw.copy()

# classify: future = wind_fcst; historical = price_da, wind_lag1, bloomberg
future_cols = ["wind_fcst"]
hist_cols   = ["price_da", "wind_lag1", "bloomberg"]

# shift y and future covariates
target_shifted["y"] = target_shifted["y"].shift(-S)
for c in future_cols:
    target_shifted[c] = target_shifted[c].shift(-S)

# drop NaN tail
target_shifted = target_shifted.dropna(subset=["y"] + future_cols).copy()
target_shifted = target_shifted.reset_index(drop=True)

print(target_shifted.to_string(index=False))
print(f"\nUsable rows after target shift: {len(target_shifted)}")
print(f"  training portion: rows 0..{TRAIN_H-1}  ({TRAIN_H} hours)")
print(f"  validation portion: rows {TRAIN_H}..{len(target_shifted)-1}  ({len(target_shifted)-TRAIN_H} hours)")
print()

# Show what each row "means" physically
print("Physical interpretation of selected rows after target shift:")
print("  Row  ds (model time)     y value   physical target time")
for i in [0, 5, 13, 14, 17]:
    if i < len(target_shifted):
        r = target_shifted.iloc[i]
        phys = r["ds"] + pd.Timedelta(hours=S)
        print(f"  {i:3d}  {r['ds']}   y={r['y']:5.0f}     → physical {phys}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Circular shift  (per-window callback, train & val shifted separately)
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print(f"3. AFTER CIRCULAR SHIFT  (K = {K_DAYS} day = {K_H} hours)")
print("   Applied to ALL non-calendar, non-meta, non-target columns.")
print("   Training and validation portions shifted SEPARATELY.")
print(SEPARATOR)

# Columns that get circularly shifted (everything except ds, unique_id, y, calendar)
# Calendar cols would be: hour, day, month, is_weekend, is_holiday, is_workday
# We don't have them in this toy example, so all feature cols get shifted.
circ_cols = future_cols + hist_cols
print(f"Circularly shifted columns: {circ_cols}\n")

double_shifted = target_shifted.copy()

# --- Shift training portion (rows 0..TRAIN_H-1) ---
T_train = TRAIN_H
K_train = K_H % T_train
print(f"Training block: {T_train} rows, K_eff = {K_H} % {T_train} = {K_train}")

train_indices = np.arange(T_train, dtype=np.intp)
train_src     = (train_indices + K_train) % T_train

for col in circ_cols:
    col_idx = double_shifted.columns.get_loc(col)
    vals = double_shifted.iloc[:TRAIN_H, col_idx].to_numpy().copy()
    double_shifted.iloc[:TRAIN_H, col_idx] = vals[train_src]

# --- Shift validation portion (rows TRAIN_H..end) ---
val_len = len(double_shifted) - TRAIN_H
K_val   = K_H % val_len if val_len > 0 else 0
print(f"Validation block: {val_len} rows, K_eff = {K_H} % {val_len} = {K_val}")

val_indices = np.arange(val_len, dtype=np.intp)
val_src     = (val_indices + K_val) % val_len

for col in circ_cols:
    col_idx = double_shifted.columns.get_loc(col)
    vals = double_shifted.iloc[TRAIN_H:, col_idx].to_numpy().copy()
    double_shifted.iloc[TRAIN_H:, col_idx] = vals[val_src]

print()
print(double_shifted.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("4. SIDE-BY-SIDE: what the model sees at each row")
print("   Columns: row | ds | y (unchanged) | cov_original → cov_after_both_shifts")
print(SEPARATOR)

compare = pd.DataFrame({
    "row":    range(len(target_shifted)),
    "ds":     target_shifted["ds"].values,
    "y":      target_shifted["y"].values,
    "split":  ["TRAIN"] * TRAIN_H + ["VAL"] * (len(target_shifted) - TRAIN_H),
})

# Show one future cov and one historical cov
for col, label in [("wind_fcst", "future"), ("bloomberg", "hist")]:
    compare[f"{col}_orig"]    = target_shifted[col].values
    compare[f"{col}_shifted"] = double_shifted[col].values
    compare[f"{col}_src_row"] = ""

    # Compute source row for training block
    for i in range(TRAIN_H):
        src = int(train_src[i])
        compare.loc[i, f"{col}_src_row"] = f"train[{src}]"

    # Compute source row for validation block
    for i in range(val_len):
        src = int(val_src[i])
        compare.loc[TRAIN_H + i, f"{col}_src_row"] = f"val[{src}]"

print(compare.to_string(index=False))
print()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Key observations
# ─────────────────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("5. KEY OBSERVATIONS - why the two shifts do NOT cancel each other")
print(SEPARATOR)
print("""
The target shift and the circular shift operate on DIFFERENT axes:

  Target shift (S = {S}h)
  ────────────────────────
  Moves along the VALUE axis: y[t] becomes y[t+S], and future covariates
  follow suit.  This means the model at model-time t learns to predict the
  physical target at time t+S.  The ds timestamp stays at t.

  Circular shift (K = {K_H}h)
  ─────────────────────────────
  Moves along the INDEX axis:  shifted[t] = original[(t + K) mod T].
  Index 0 now holds original[K], index K holds original[2K], etc.
  (Left-rotation by K steps.)
  This scrambles the temporal alignment between covariates and the target
  WITHOUT changing what y the model predicts.

Combined effect at row t (with K < T, so K_eff = K):
  • y[t]         = original y at physical time (t + S)        ← target shift
  • hist_cov[t]  = original hist_cov at time (t + K) mod T   ← circular shift only
  • futr_cov[t]  = original futr_cov at time (t + K) mod T   ← circular shift
                   (these were already target-shifted to physical time t+S
                    BEFORE the circular shift permuted the indices)

The circular shift does NOT undo the target shift because:
  1. The target shift changed WHICH VALUE sits at position t.
  2. The circular shift then MOVES that already-target-shifted value
     to a different index.
  3. y is NEVER circularly shifted - only covariates are permuted.

So the model still predicts the correct future y (aligned to physical time
t+S), but the covariate vector it receives at row t no longer corresponds
to time t or time t+S - it comes from time (t + K) mod T, a completely
different point in the training cycle.

If the model performs as well on the test set despite this scrambling,
it means the temporal covariate patterns were not contributing to the
forecast - the covariates were effectively a placebo.
""".format(S=S, K_H=K_H))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Example: trace one specific row in detail
# ─────────────────────────────────────────────────────────────────────────────
TRACE_ROW = 5
print(SEPARATOR)
print(f"6. DETAILED TRACE - training row {TRACE_ROW}")
print(SEPARATOR)
r_raw = raw.iloc[TRACE_ROW]
r_ts  = target_shifted.iloc[TRACE_ROW]
r_ds  = double_shifted.iloc[TRACE_ROW]
src_train = int(train_src[TRACE_ROW])
src_ts = target_shifted.iloc[src_train]

print(f"""
Row {TRACE_ROW} in the RAW data:
  ds = {r_raw['ds']}  y = {r_raw['y']}  wind_fcst = {r_raw['wind_fcst']}  bloomberg = {r_raw['bloomberg']}

After TARGET SHIFT (S={S}):
  ds = {r_ts['ds']}  (unchanged)
  y  = {r_ts['y']}   (was raw y at t+{S} = row {TRACE_ROW+S}, raw value = {raw.iloc[TRACE_ROW+S]['y']})
  wind_fcst = {r_ts['wind_fcst']}  (was raw wind_fcst at t+{S} = {raw.iloc[TRACE_ROW+S]['wind_fcst']})
  bloomberg = {r_ts['bloomberg']}  (historical - NOT target-shifted, same as raw)

After CIRCULAR SHIFT (K_eff={K_train} within training block of {T_train} rows):
  Source row for covariates: ({TRACE_ROW} + {K_train}) mod {T_train} = {src_train}
  ds = {r_ds['ds']}  (unchanged - never shifted)
  y  = {r_ds['y']}   (unchanged - never circularly shifted)
  wind_fcst = {r_ds['wind_fcst']}  (was target-shifted wind_fcst at row {src_train} = {src_ts['wind_fcst']})
  bloomberg = {r_ds['bloomberg']}  (was bloomberg at row {src_train} = {src_ts['bloomberg']})

Summary for row {TRACE_ROW}:
  Model predicts:  y from physical time {r_ts['ds'] + pd.Timedelta(hours=S)}
  Using future cov from: physical time {raw.iloc[src_train + S]['ds']} (target-shifted wind_fcst)
  Using hist cov from:   physical time {raw.iloc[src_train]['ds']} (bloomberg)
  → The future covariate is from {abs(TRACE_ROW + S - (src_train + S))}h away from the correct time
  → The historical covariate is from {abs(TRACE_ROW - src_train)}h away from the correct time
  → The covariates are MISALIGNED with the target - this is the placebo effect
""")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Prediction context: what happens at test time
# ─────────────────────────────────────────────────────────────────────────────
L = 4  # input_size (lookback)
print(SEPARATOR)
print(f"7. PREDICTION CONTEXT - first stride (input_size L = {L})")
print(SEPARATOR)
print(f"""
At test time, predict_with_shift_correction draws context from the CLEAN
(only target-shifted) dataset - NOT the circularly shifted one.

The last L = {L} rows before the test period come from the end of the
validation portion, with REAL (unshifted) covariates:
""")

# The validation rows in the target-shifted df
val_rows = target_shifted.iloc[TRAIN_H:]
last_L = val_rows.tail(L)

print("Lookback context (from clean target-shifted df, last L rows of val):")
print(last_L.to_string(index=False))
print()
print("During training, these same rows had SHIFTED covariates:")
print(double_shifted.iloc[TRAIN_H:].tail(L).to_string(index=False))
print(f"""
This is a deliberate distribution mismatch:
  • Training:   model saw shifted covariates (temporal patterns broken)
  • Prediction: model sees real covariates (temporal patterns intact)

If the model learned to exploit temporal patterns in the covariates,
it will benefit from real data at test time → performance may be BETTER
than during training.  If it ignored covariates (learned only from y),
the switch won't matter.  Either way, there is no unfair advantage:
the test set evaluation is identical to the non-augmented baseline.
""")

print(SEPARATOR)
print("DONE")
print(SEPARATOR)
