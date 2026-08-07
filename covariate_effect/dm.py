import pandas as pd
import numpy as np
from scipy import stats

def diebold_mariano_test(actual, forecast1, forecast2, horizon=1):
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    H0: forecast1 and forecast2 have equal accuracy
    H1: forecasts have different accuracy
    
    Returns: test_statistic, p_value
    """
    # Calculate forecast errors
    e1 = actual - forecast1
    e2 = actual - forecast2
    
    # Loss differential (using squared error)
    d = e1**2 - e2**2
    
    # Mean loss differential
    mean_d = d.mean()
    
    # Variance of loss differential (with Newey-West adjustment for autocorrelation)
    n = len(d)
    var_d = d.var()
    
    # Newey-West lag = horizon - 1
    lag = horizon - 1
    if lag > 0:
        for k in range(1, lag + 1):
            cov_k = np.cov(d[:-k], d[k:])[0, 1]
            var_d += 2 * (1 - k / (lag + 1)) * cov_k
    
    # DM test statistic
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def compare_models(predictions1_df: pd.DataFrame, predictions2_df: pd.DataFrame, start_date: str | None = None):
    results = []

    common_models = (
        set(predictions1_df.columns).intersection(set(predictions2_df.columns))
        .difference({"unique_id", "tso", "y", "ds", "horizon"})
    )
    for tso, direction in predictions1_df[["tso", "unique_id"]].drop_duplicates().itertuples(index=False):
        tso_data1 = predictions1_df[np.logical_and(
            predictions1_df["tso"] == tso,
            predictions1_df["unique_id"] == direction
        )]
        if start_date is not None:
            tso_data1 = tso_data1[tso_data1["ds"] >= start_date]
        tso_data2 = predictions2_df[np.logical_and(
            predictions2_df["tso"] == tso,
            predictions2_df["unique_id"] == direction
        )]
        if start_date is not None:
            tso_data2 = tso_data2[tso_data2["ds"] >= start_date]
        if tso_data1.empty or tso_data2.empty:
            continue
        actual = tso_data1["y"].reset_index(drop=True)
        for model_col in common_models:
            forecast1 = tso_data1[model_col].reset_index(drop=True)
            forecast2 = tso_data2[model_col].reset_index(drop=True)
            dm_stat, p_value = diebold_mariano_test(actual, forecast1, forecast2, horizon=24)
            results.append({
                "tso": tso,
                "direction": direction,
                "model": model_col,
                "dm_statistic": dm_stat,
                "p_value": p_value
            })

    return pd.DataFrame(results)


def pairwise_dm_comparison(predictions_df: pd.DataFrame, start_date: str | None = None):
    """
    Perform pairwise Diebold-Mariano tests between all models in predictions_df.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame containing predictions with columns for each model, plus metadata columns
        (unique_id, ds, horizon, tso, window_index, y)
    start_date : str | None
        Optional start date to filter the data
        
    Returns:
    --------
    pd.DataFrame
        A matrix with models as both rows and columns, filled only in upper triangle.
        Each cell contains the DM statistic (positive means row model is better than column model).
    """
    # Identify metadata columns to skip
    metadata_cols = {'unique_id', 'ds', 'horizon', 'tso', 'window_index', 'y'}
    
    # Get model columns
    model_cols = [col for col in predictions_df.columns if col not in metadata_cols]
    
    # Filter by start date if provided
    df = predictions_df.copy()
    if start_date is not None:
        df = df[df['ds'] >= start_date]
    
    # Initialize result matrix with NaN
    n_models = len(model_cols)
    dm_matrix = pd.DataFrame(
        np.nan, 
        index=model_cols, 
        columns=model_cols
    )
    p_value_matrix = pd.DataFrame(
        np.nan, 
        index=model_cols, 
        columns=model_cols
    )
    
    # Get actual values
    actual = df['y'].reset_index(drop=True)
    
    # Compute pairwise DM tests (upper triangle only)
    for i, model1 in enumerate(model_cols):
        for j, model2 in enumerate(model_cols):
            if i < j:  # Upper triangle only
                forecast1 = df[model1].reset_index(drop=True)
                forecast2 = df[model2].reset_index(drop=True)
                
                # Perform DM test: positive statistic means model1 is better than model2
                dm_stat, p_value = diebold_mariano_test(actual, forecast1, forecast2, horizon=24)
                
                dm_matrix.loc[model1, model2] = dm_stat
                p_value_matrix.loc[model1, model2] = p_value
            elif i == j:
                # Diagonal is 0 (comparing model to itself)
                dm_matrix.loc[model1, model2] = 0.0
                p_value_matrix.loc[model1, model2] = 1.0
    
    return dm_matrix, p_value_matrix


def show_dm_dominance(comparison_df: pd.DataFrame, target: pd.DataFrame, sparsity_threshold: float = 0.7, alpha: float = 0.05, lower_better: bool = False,):
    df = comparison_df.copy()
    sparsity = target.groupby(["tso", "unique_id"])["y"].apply(lambda g: (g == 0.).mean())
    df = df.rename(columns={"direction": "unique_id"}).merge(sparsity.rename("sparsity"), on=["tso", "unique_id"], how="left")
    df["is_sparse"] = np.where(df["sparsity"] >= sparsity_threshold, "Sparse", "Dense")
    n_rows_per_sparsity = df.groupby(["model", "is_sparse"])["unique_id"].count()
    dominance_neg = df.groupby(["model", "is_sparse"])[["p_value", "dm_statistic"]].apply(
        lambda row: ((row["p_value"] < alpha) & (row["dm_statistic"] < 0 if lower_better else row["dm_statistic"] > 0)).sum()
    ).reset_index(name="Expected")
    dominance_pos = df.groupby(["model", "is_sparse"])[["p_value", "dm_statistic"]].apply(
        lambda row: ((row["p_value"] < alpha) & (row["dm_statistic"] > 0 if lower_better else row["dm_statistic"] < 0)).sum()
    ).reset_index(name="Unexpected")
    dominance = dominance_neg.merge(dominance_pos, on=["model", "is_sparse"], how="outer")
    dominance = dominance.set_index(["model", "is_sparse"]).div(n_rows_per_sparsity, axis=0)
    return dominance.reset_index()