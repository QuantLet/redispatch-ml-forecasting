import pandas as pd
import numpy as np

from numba import njit
from sklearn.metrics import fbeta_score
from tqdm.auto import tqdm

from training.runner import _compute_rolling_windows

@njit(cache=True)
def block_bootstrap_jit(n: int, block_len: int) -> np.ndarray:
    bootstrap_indices = np.empty(n, dtype=np.int64)

    pos = 0
    while pos < n:
        bidx = np.random.randint(0, n - block_len + 1)
        start = bidx
        end = start + block_len

        for j in range(start, end):
            bootstrap_indices[pos] = j
            pos += 1
            if pos == n:
                break

    return bootstrap_indices


def block_bootstrap_fbeta_jit(
    data: pd.DataFrame, 
    block_len: int = 24, 
    n_bootstrap: int = 1000, 
    beta: float = 2.0,
) -> pd.DataFrame:
    results = []
    model_cols = [
        col for col in data.columns 
        if col not in ["unique_id", "tso", "horizon", "window_index", "ds", "y"]
    ]
    data_arr = data.copy().sort_values("ds")[model_cols + ["y"]].to_numpy(dtype="float64")
    block_bootstrap_jit(20, 5)  # warmup jit compilation
    for _ in range(n_bootstrap):
        bootstrap_indices = block_bootstrap_jit(len(data), block_len)
        bootstrap_sample = data_arr[bootstrap_indices, :]
        y_true = (bootstrap_sample[:, -1] > 0).astype(int)
        per_model_fbeta = pd.Series({
            model: fbeta_score(
                y_true, (bootstrap_sample[:, i] > 0).astype(int), 
                average='binary', zero_division=0, beta=beta
            )
            for i, model in enumerate(model_cols)
        })
        fbeta_deltas = per_model_fbeta - per_model_fbeta.max()
        results.append({
            "unique_id": data["unique_id"].iloc[0],
            "tso": data["tso"].iloc[0],
            **{f"{model}_delta": delta for model, delta in fbeta_deltas.items()}
        })
            
    return pd.DataFrame(results)


def bootstrap_paired_test_jit(
    data: pd.DataFrame, 
    block_len: int = 24, 
    n_bootstrap: int = 1000, 
    beta: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cols = [
        col for col in data.columns 
        if col not in ["unique_id", "tso", "horizon", "window_index", "ds", "y"]
    ]
    wins = np.zeros((len(model_cols), len(model_cols)), dtype=np.int64)
    ties = np.zeros((len(model_cols), len(model_cols)), dtype=np.int64)
    deltas = []
    data_arr = data.copy().sort_values("ds")[model_cols + ["y"]].to_numpy(dtype="float64")
    block_bootstrap_jit(20, 5)  # warmup jit compilation
    for _ in range(n_bootstrap):
        bootstrap_indices = block_bootstrap_jit(len(data), block_len)
        bootstrap_sample = data_arr[bootstrap_indices, :]
        y_true = (bootstrap_sample[:, -1] > 0).astype(int)
        per_model_fbeta = np.array([
            fbeta_score(
                y_true, (bootstrap_sample[:, i] > 0).astype(int), 
                average='binary', zero_division=0, beta=beta
            )
            for i in range(len(model_cols))
        ])
        for i in range(len(model_cols)):
            for j in range(len(model_cols)):
                if per_model_fbeta[i] > per_model_fbeta[j]:
                    wins[i, j] += 1
                elif per_model_fbeta[i] == per_model_fbeta[j]:
                    ties[i, j] += 1
        model_deltas = pd.Series(per_model_fbeta - per_model_fbeta.max(), index=model_cols)
        deltas.append({
            "unique_id": data["unique_id"].iloc[0],
            "tso": data["tso"].iloc[0],
            **{f"{model}_delta": delta for model, delta in model_deltas.items()}
        })
            
    superiority_tests = pd.DataFrame((wins + ties / 2) / n_bootstrap, columns=model_cols, index=model_cols)
    return superiority_tests, pd.DataFrame(deltas)


def bootstrap_acceptance_rule(simulations: pd.DataFrame, eps: float = 0.05, alpha: float = 0.05) -> pd.DataFrame:
    acceptance = []
    for (tso, unique_id), group in simulations.groupby(["tso", "unique_id"]):
        model_delta_cols = [col for col in group.columns if col.endswith("_delta")]
        acceptance_counts = (group[model_delta_cols] >= -eps).sum()
        acceptance_rates = acceptance_counts / len(group) >= 1 - alpha
        acceptance.append({
            "tso": tso,
            "unique_id": unique_id,
            **{model: acceptance_rates[model] for model in model_delta_cols}
        })
    return pd.DataFrame(acceptance)


def run_bootstrap_fbeta_tests(
    data: pd.DataFrame, 
    block_len: int = 24, 
    n_bootstrap: int = 1000, 
    beta: float = 2.0,
    eps_acceptance: float = 0.02,
    alpha_acceptance: float = 0.1,
    show_progress: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    test_probs_dict = {}
    acceptance_list = []
    with tqdm(
        total=data["tso"].nunique() * data["unique_id"].nunique(), 
        desc="Running bootstrap tests", 
        disable=not show_progress
    ) as pbar:
        for (tso, unique_id), group in data.groupby(["tso", "unique_id"]):
            pbar.set_postfix(tso=tso, unique_id=unique_id)
            test_probs, deltas = bootstrap_paired_test_jit(
                group, block_len=block_len, n_bootstrap=n_bootstrap, beta=beta
            )
            test_probs_dict[(tso, unique_id)] = test_probs
            acceptance = bootstrap_acceptance_rule(deltas, eps=eps_acceptance, alpha=alpha_acceptance)
            acceptance_list.append(acceptance)
            pbar.update(1)

    return test_probs_dict, pd.concat(acceptance_list, ignore_index=True)


def per_window_fbeta_statistics(
    preds_df: pd.DataFrame,
    start_date: pd.Timestamp,
    rolling_window_n_train_months: int = 37,
    rolling_window_n_valid_months: int = 2,
    rolling_window_n_test_months: int = 1,
    block_len: int = 24, 
    n_bootstrap: int = 1000, 
    beta: float = 2.0,
):
    rolling_windows = _compute_rolling_windows(
        n_train_months=rolling_window_n_train_months,
        n_valid_months=rolling_window_n_valid_months,
        n_test_months=rolling_window_n_test_months,
        data_start=start_date,
        data_end=preds_df["ds"].max(),
    )
    all_stats = []
    all_tests = []
    with tqdm(total=preds_df["tso"].nunique() * preds_df["unique_id"].nunique() * len(rolling_windows), desc="Evaluating rolling windows") as pbar:
        for (tso, unique_id), group in preds_df.groupby(["tso", "unique_id"]):
            for window_index, wb in enumerate(rolling_windows):
                window_group = group[
                    group["ds"].between(wb.valid_start, wb.test_start - pd.Timedelta(hours=1))
                ].copy()
                expected_date_range = pd.date_range(start=wb.valid_start, end=wb.test_start - pd.Timedelta(hours=1), freq="h")
                if window_index < len(rolling_windows) - 1 and not window_group["ds"].isin(expected_date_range).all():
                    raise ValueError(f"Window group for TSO {tso} and unique_id {unique_id} and window_index {window_index} does not cover the expected date range. Missing dates: {expected_date_range[~expected_date_range.isin(window_group['ds'])]}")
                if window_group.empty:
                    continue
                pbar.set_postfix(current=f"{tso} {unique_id} w{window_index}", w_start=wb.valid_start.date(), w_end=wb.test_start.date())
                test_df, stats_df = bootstrap_paired_test_jit(
                    window_group, block_len=block_len, n_bootstrap=n_bootstrap, beta=beta
                )
                stats_df["window_index"] = window_index
                test_df["window_index"] = window_index
                all_stats.append(stats_df)
                all_tests.append(test_df)
                pbar.update(1)
    return pd.concat(all_stats, ignore_index=True), pd.concat(all_tests, ignore_index=True)