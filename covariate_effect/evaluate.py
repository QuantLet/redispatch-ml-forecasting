import pandas as pd

from sklearn.metrics import fbeta_score
from training.predict import evaluate_models

def classification_metrics_df(prepared_preds_df, id_col='unique_id', time_col='ds', target_col='y', beta: float = 1.0):
    # basic validation
    if id_col not in prepared_preds_df.columns or time_col not in prepared_preds_df.columns or target_col not in prepared_preds_df.columns:
        raise ValueError(f"prepared_preds_df must contain columns {id_col}, {time_col}, {target_col}")

    # choose model columns automatically if not provided
    exclude = {id_col, time_col, target_col}
    models = [c for c in prepared_preds_df.columns if c not in exclude]
    if len(models) == 0:
        raise ValueError("No model columns found in prepared_preds_df")

    # directions to evaluate
    directions = prepared_preds_df[id_col].unique()

    results = []
    for direction in directions:
        sub = prepared_preds_df[prepared_preds_df[id_col] == direction]
        # ground truth binary
        y_true = (sub[target_col] > 0).astype(int)
        result_row = {"unique_id": direction, "metric": f"fbeta_{beta}"}
        for model in models:
            y_pred = (sub[model] > 0).astype(int)
            fbeta = fbeta_score(y_true, y_pred, average='binary', zero_division=0, beta=beta)
            result_row[model] = fbeta
        results.append(result_row)
    return pd.DataFrame(results)

def evaluation_pipeline(pivoted_group: pd.DataFrame, beta: float = 1.0) -> pd.DataFrame:
    evaluted_group_overall = evaluate_models(pivoted_group)
    evaluted_group_overall_to_keep = evaluted_group_overall[evaluted_group_overall["metric"] == "mae"].copy()
    evaluated_conditional_group = evaluate_models(pivoted_group[pivoted_group["y"] > 0])
    evaluated_conditional_group_to_keep = evaluated_conditional_group[evaluated_conditional_group["metric"] == "mae"].copy()
    evaluated_conditional_group_to_keep["metric"] = evaluated_conditional_group_to_keep["metric"].map({"mae": "mae_conditional"})
    evaluated_fbeta_group = classification_metrics_df(pivoted_group, beta=beta)
    evaluated_fbeta_group_to_keep = evaluated_fbeta_group[evaluated_fbeta_group["metric"] == f"fbeta_{beta}"].copy()
    r2_score = evaluted_group_overall[evaluted_group_overall["metric"] == "r2_score"].copy()
    metrics_to_concatenate = [evaluted_group_overall_to_keep, evaluated_conditional_group_to_keep, evaluated_fbeta_group_to_keep, r2_score]
    evaluted_group = pd.concat(metrics_to_concatenate, ignore_index=True)
    return evaluted_group

def perform_tso_evaluations_rolling_window(preds_df: pd.DataFrame, beta: float = 2.0, start_date: pd.Timestamp | str | None = None, end_date: pd.Timestamp | str | None = None) -> pd.DataFrame:
    evaluation_results = []
    if start_date is not None:
        preds_df = preds_df[preds_df["ds"] >= start_date]
    if end_date is not None:
        preds_df = preds_df[preds_df["ds"] <= end_date]
    for tso, group in preds_df.groupby("tso"):
        evaluated_group = evaluation_pipeline(group.drop(columns=["tso", "window_index", "horizon"], errors="ignore"), beta=beta)
        sparsity = group.groupby("unique_id")["y"].apply(lambda x: (x == 0).mean()).reset_index(name="sparsity")
        evaluated_group = evaluated_group.merge(sparsity, on="unique_id", how="left")
        volume = group.groupby("unique_id")["y"].sum().reset_index(name="volume")
        evaluated_group = evaluated_group.merge(volume, on="unique_id", how="left")
        evaluated_group["tso"] = tso
        evaluation_results.append(evaluated_group)
    return pd.concat(evaluation_results, ignore_index=True)