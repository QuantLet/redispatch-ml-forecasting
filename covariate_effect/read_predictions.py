import pandas as pd
import numpy as np
import re
import time
import wandb
import tempfile
import threading

from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from training.predict import prepare_predictions_df
from training.data_prep import load_dataset, to_nixtla_format
from covariate_effect.constants import TSO_NAMES, TSO_NAME_MAPPING

def retrieve_model_predictions_from_wandb(
    wandb_entity: str | None,
    wandb_project: str,
    model_name: str,
    tso_name: str,
    timestamp: str,
    best_checkpoint: bool,
    window_index: int | None = None,
    api: wandb.Api | None = None,
    download_dir: str | Path | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    fetch_file_only: bool = False
) -> pd.DataFrame:
    """Retrieve predictions artifact logged by training.predict.log_predictions_to_wandb."""
    if window_index is not None:
        if best_checkpoint:
            checkpoint_label = f"test_best_valid_window{window_index}"
        else:
            checkpoint_label = f"test_window{window_index}"
    else:
        if best_checkpoint:
            checkpoint_label = "test_best_valid"
        else:
            checkpoint_label = "test"
    artifact_name = f"preds_{checkpoint_label}_{model_name}_{tso_name}_{timestamp}"
    project_path = f"{wandb_entity}/{wandb_project}" if wandb_entity else wandb_project
    _api = api if api is not None else wandb.Api()
    _download_dir = str(download_dir) if download_dir is not None else "wandb"
    filename = f"{checkpoint_label}_predictions.parquet"

    for attempt in range(max_retries + 1):
        try:
            predictions_artifact = _api.artifact(f"{project_path}/{artifact_name}:latest")
            if fetch_file_only:
                # Fetch only the single file of interest (no full artifact manifest download).
                # skip_cache=True writes directly to our per-thread dest dir, bypassing
                # wandb's global shared file cache and avoiding cross-thread contention.
                predictions_path = Path(
                    predictions_artifact.get_entry(filename).download(root=_download_dir, skip_cache=True)
                )
                if not predictions_path.exists():
                    raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
            else:
                # Full artifact download (including manifest). This is more robust to changes in how files are logged
                # (e.g. if we switch to logging a directory of predictions instead of a single file), but may be
                # slower due to the manifest download and potential extra file downloads.
                predictions_path = Path(predictions_artifact.download(root=_download_dir)) / filename
                if not predictions_path.exists():
                    raise FileNotFoundError(f"Predictions file not found at {predictions_path}")
            result = prepare_predictions_df(
                pd.read_parquet(predictions_path),
                actuals_df=pd.DataFrame()
            )
            if result.empty:
                raise ValueError("prepare_predictions_df returned an empty DataFrame")
            return result
        except Exception as e:
            if attempt == max_retries:
                return pd.DataFrame()
            is_rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
            # Exponential backoff; longer initial wait for rate-limit errors
            wait = retry_delay * (4 ** attempt) if is_rate_limited else retry_delay * (2 ** attempt)
            time.sleep(wait)

    return pd.DataFrame()

def obtain_predictions(
    root_dir: Path, 
    wandb_project: str, 
    rolling_window: bool, 
    best_checkpoint: bool, 
    dataset_name: str,
    wandb_entity: str | None = None,
    model_filter: str = "seed"
):
    normalized_tso_names = [tso.lower() for tso in TSO_NAMES]

    def get_timestamp_from_dir_name(model_path: Path) -> str:
        timestmap_pattern_regex = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        dir_name = model_path.name
        match = re.search(timestmap_pattern_regex, dir_name)
        if match:
            timestamp_str = match.group(0)
            return timestamp_str
        files_in_dir = list(model_path.glob("*.tar.zst"))
        if files_in_dir:
            match = [file for file in files_in_dir if re.search(timestmap_pattern_regex, file.name)]
            if match:
                # Choose the latest timestamp if multiple matches are found
                match.sort(key=lambda x: re.search(timestmap_pattern_regex, x.name).group(0), reverse=True)
                if len(match) > 1:
                    print(f"Multiple timestamp matches found in {model_path}, using the latest one: {match[0].name}")
                return match[0].name.split(".tar.zst")[0]  # Return just the timestamp part without extension
            else:
                raise ValueError(f"No timestamp found in files of directory: {dir_name}")
        else:
            raise ValueError(f"No timestamp found in directory name: {dir_name}")
        
    def convert_timestamp_to_datetime(timestamp: str) -> pd.Timestamp:
        dt = pd.to_datetime(timestamp, format="%Y-%m-%d_%H-%M-%S")
        return dt
    
    def merge_tso_data(predictions_list: list[pd.DataFrame]) -> pd.DataFrame:
        merge_cols = ["unique_id", "ds", "tso", "horizon", "y"]
        if "window_index" in predictions_list[0].columns:
            merge_cols.append("window_index")
        if predictions_list:
            result = predictions_list[0]
            for to_merge in predictions_list[1:]:
                result = pd.merge(result, to_merge, on=merge_cols, how="outer")
        else:
            result = pd.DataFrame()
        return result

    predictions_list = []
    if rolling_window:
        for tso_dir in root_dir.iterdir():
            if tso_dir.is_dir() and tso_dir.name.lower() in normalized_tso_names:
                tso_dataset_dir = tso_dir / dataset_name
                if not tso_dataset_dir.exists():
                    print(f"Dataset directory {tso_dataset_dir} does not exist.")
                    continue
                for window_dir in tso_dataset_dir.iterdir():
                    if "window_" not in window_dir.name:
                        continue
                    window_index = int(window_dir.name.split("window_")[-1])
                    for model_dir in window_dir.iterdir():
                        if model_dir.is_dir() and model_filter in model_dir.name:
                            model_name = model_dir.name
                        else:
                            continue
                        timestamp = get_timestamp_from_dir_name(model_dir)
                        tso_name = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
                        model_predictions = retrieve_model_predictions_from_wandb(
                            wandb_entity=wandb_entity,
                            wandb_project=wandb_project,
                            model_name=model_name,
                            tso_name=tso_name,
                            timestamp=timestamp,
                            best_checkpoint=best_checkpoint,
                            window_index=window_index
                        )
                        if not model_predictions.empty:
                            # model_predictions["model"] = model_name
                            model_predictions["tso"] = tso_name
                            model_predictions["window_index"] = window_index
                            predictions_list.append(model_predictions)
    else:
        search_root_dir = root_dir / dataset_name
        if not search_root_dir.exists():
            print(f"Search root directory {search_root_dir} does not exist.")
            return pd.DataFrame()
        for tso_dir in search_root_dir.iterdir():
            if tso_dir.is_dir() and tso_dir.name.lower() in normalized_tso_names:
                latest_subdir = max(tso_dir.iterdir(), key=lambda d: convert_timestamp_to_datetime(get_timestamp_from_dir_name(d)))
                print(f"Latest subdirectory for {tso_dir.name}: {latest_subdir}")
                for model_dir in latest_subdir.iterdir():
                    if model_dir.is_dir() and model_filter in model_dir.name:
                        model_name = model_dir.name
                    else:
                        continue
                    model_predictions = retrieve_model_predictions_from_wandb(
                        wandb_entity=wandb_entity,
                        wandb_project=wandb_project,
                        model_name=model_name,
                        tso_name=TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name),
                        timestamp=get_timestamp_from_dir_name(latest_subdir),
                        best_checkpoint=best_checkpoint,
                    )
                    if not model_predictions.empty:
                        # model_predictions["model"] = model_name
                        model_predictions["tso"] = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
                        predictions_list.append(model_predictions)

    if predictions_list:
        if rolling_window:
            per_tso_predictions_list = []
            for window_index in sorted(set(pred["window_index"].iloc[0] for pred in predictions_list)):
                window_predictions = [pred for pred in predictions_list if pred.get("window_index", -1).iloc[0] == window_index]
                if window_predictions:
                    for tso in TSO_NAME_MAPPING.values():
                        tso_predictions = [pred for pred in window_predictions if pred["tso"].iloc[0] == tso]
                        if tso_predictions:
                            merged_tso_predictions = merge_tso_data(tso_predictions)
                            per_tso_predictions_list.append(merged_tso_predictions)
            result = pd.concat(per_tso_predictions_list, ignore_index=True) if per_tso_predictions_list else pd.DataFrame()
        else:
            per_tso_predictions_list = []
            for tso in TSO_NAME_MAPPING.values():
                tso_predictions = [pred for pred in predictions_list if pred["tso"].iloc[0] == tso]
                if tso_predictions:
                    merged_tso_predictions = merge_tso_data(tso_predictions)
                    per_tso_predictions_list.append(merged_tso_predictions)
            result = pd.concat(per_tso_predictions_list, ignore_index=True) if per_tso_predictions_list else pd.DataFrame()
    else:
        result = pd.DataFrame()

    return result


def read_benchmark_predictions(root_dir: Path, dataset_name: str, start_date: pd.Timestamp | None = None) -> pd.DataFrame:
    benchmark_dfs = []
    for tso_dir in root_dir.iterdir():
        if tso_dir.is_dir() and tso_dir.name.lower() in [tso.lower() for tso in TSO_NAMES]:
            benchmark_dir = tso_dir / dataset_name / "evaluation"
            if benchmark_dir.exists():
                benchmark_file_path = next(benchmark_dir.glob("benchmarks*.csv"), None)
                if benchmark_file_path:
                    df = pd.read_csv(benchmark_file_path, parse_dates=["ds"])
                    df["tso"] = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
                    df = df.drop(columns=["dataset"])
                    df["horizon"] = df.groupby(["tso", "unique_id", df["ds"].dt.normalize()]).cumcount() + 1
                    if start_date is not None:
                        df = df[df["ds"] >= start_date]
                    df.columns = df.columns.str.replace("linear_regression", "ridge_regression")
                    benchmark_dfs.append(df)
                else:
                    print(f"Benchmark file {benchmark_file_path} does not exist.")
    if benchmark_dfs:
        return pd.concat(benchmark_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def obtain_predictions_multithreaded(
    root_dir: Path,
    wandb_project: str,
    rolling_window: bool,
    best_checkpoint: bool,
    dataset_name: str,
    wandb_entity: str | None = None,
    model_filter: str = "seed",
):
    
    def _extract_timestamp(model_path: Path) -> str:
        rx = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        hit = re.search(rx, model_path.name)
        if hit:
            return hit.group(0)

        archive_hits = sorted(
            [p for p in model_path.glob("*.tar.zst") if re.search(rx, p.name)],
            key=lambda p: re.search(rx, p.name).group(0),
            reverse=True,
        )
        if archive_hits:
            return archive_hits[0].name.replace(".tar.zst", "")

        raise ValueError(f"No timestamp found for: {model_path}")


    # Each worker thread gets its own wandb.Api instance and a private temp directory,
    # fully isolating threads from wandb's global state (wandb/wandb#6856).
    _thread_local = threading.local()

    def _get_thread_context() -> tuple[wandb.Api, str]:
        """Return a (Api, download_dir) pair that is private to the calling thread."""
        if not hasattr(_thread_local, "api"):
            _thread_local.api = wandb.Api()
            _thread_local.tmp_dir = tempfile.mkdtemp(prefix="wandb_dl_")
        return _thread_local.api, _thread_local.tmp_dir

    if not rolling_window:
        return obtain_predictions(
            root_dir=root_dir,
            wandb_project=wandb_project,
            rolling_window=rolling_window,
            best_checkpoint=best_checkpoint,
            dataset_name=dataset_name,
            wandb_entity=wandb_entity,
            model_filter=model_filter,
        )
    else:
        allowed = {x.lower() for x in TSO_NAMES}
        jobs: list[tuple[str, str, str, int]] = []

        for tso_dir in root_dir.iterdir():
            if not tso_dir.is_dir() or tso_dir.name.lower() not in allowed:
                continue

            base = tso_dir / dataset_name
            if not base.exists():
                continue

            tso_label = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
            for wdir in base.iterdir():
                if "window_" not in wdir.name:
                    continue
                widx = int(wdir.name.split("window_")[-1])

                for mdir in wdir.iterdir():
                    if not (mdir.is_dir() and model_filter in mdir.name):
                        continue
                    jobs.append((mdir.name, tso_label, _extract_timestamp(mdir), widx))

        if not jobs:
            return pd.DataFrame()

        def _fetch(job: tuple[str, str, str, int]) -> pd.DataFrame:
            model_name, tso_label, ts, widx = job
            thread_api, thread_dir = _get_thread_context()
            pred = retrieve_model_predictions_from_wandb(
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                model_name=model_name,
                tso_name=tso_label,
                timestamp=ts,
                best_checkpoint=best_checkpoint,
                window_index=widx,
                api=thread_api,
                download_dir=thread_dir,
            )
            if pred.empty:
                return pred
            pred["tso"] = tso_label
            pred["window_index"] = widx
            pred = pred.melt(
                id_vars=["unique_id", "ds", "tso", "horizon", "y", "window_index"],
                var_name="model",
                value_name="y_pred"
            )
            return pred

        workers = min(4, len(jobs))
        gathered: list[pd.DataFrame] = []
        # Create the progress bar before entering the executor so Jupyter
        # can render the widget immediately, rather than only at the end.
        with tqdm(total=len(jobs), desc="Downloading artifacts") as pbar:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_fetch, j): j for j in jobs}
                for fut in as_completed(futures):
                    model_name, tso_label, _, widx = futures[fut]
                    pbar.set_postfix_str(f"{tso_label} w{widx} {model_name[:30]}")
                    df = fut.result()
                    if not df.empty:
                        gathered.append(df)
                    pbar.update(1)

        return pd.concat(gathered, ignore_index=True) if gathered else pd.DataFrame()

def obtain_predictions_from_explainability_results(
    root_dir: Path,
    root_dataset_dir: Path,
    dataset_name: str,
    model_filter: str = "seed",
):
    """Predictions can be obtained from the `evaluation/ig_preds` directory for each window, but only for the last checkpoint."""
    all_predictions_list = []

    for tso_dir in root_dir.iterdir():
        tso_full_name = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
        actuals_df, _ = load_dataset(root_dataset_dir / (dataset_name + f"_{tso_full_name}.parquet"))
        actuals_df = to_nixtla_format(actuals_df)
        if tso_dir.is_dir() and tso_dir.name.lower() in [tso.lower() for tso in TSO_NAMES]:
            tso_dataset_dir = tso_dir / dataset_name
            if not tso_dataset_dir.exists():
                print(f"Dataset directory {tso_dataset_dir} does not exist.")
                continue
            for window_dir in tso_dataset_dir.iterdir():
                if "window_" not in window_dir.name:
                    continue
                window_index = int(window_dir.name.split("window_")[-1])
                ig_preds_dir = window_dir / "evaluation" / "ig_preds"
                if not ig_preds_dir.exists():
                    print(f"IG predictions directory {ig_preds_dir} does not exist.")
                    continue
                for pred_file in ig_preds_dir.glob("*.parquet"):
                    if model_filter in pred_file.name:
                        df = pd.read_parquet(pred_file)
                        df = prepare_predictions_df(df, actuals_df=actuals_df)
                        df = df.melt(
                            id_vars=["unique_id", "ds", "horizon", "y"],
                            var_name="model",
                            value_name="y_pred"
                        )
                        df["tso"] = tso_full_name
                        df["window_index"] = window_index
                        all_predictions_list.append(df)
    
    all_predictions = pd.concat(all_predictions_list, ignore_index=True) if all_predictions_list else pd.DataFrame()
    return all_predictions.pivot_table(
        index=["tso", "unique_id", "ds", "horizon", "y"],
        columns="model",
        values="y_pred"
    ).reset_index()


def read_local_rolling_window_predictions(
    rolling_window_shift_dir: Path, 
    dataset_name: str,
    evaluation_dir_name: str = "evaluation",
    model_filter: str = "seed",
    checkpoint_best: bool = False,
) -> pd.DataFrame:
    all_dfs = []
    for tso_dir in rolling_window_shift_dir.glob("*"):
        if tso_dir.is_dir() and tso_dir.name in TSO_NAME_MAPPING.keys():
            normalized_tso_name = TSO_NAME_MAPPING.get(tso_dir.name.lower(), tso_dir.name)
            if checkpoint_best:
                window_regex = re.compile(fr"predictions_(.*)_{normalized_tso_name}_.*_window(\d+)_best_checkpoint\.parquet")
            else:
                window_regex = re.compile(fr"predictions_(.*)_{normalized_tso_name}_.*_window(\d+)\.parquet")
            eval_dir = rolling_window_shift_dir / tso_dir.name / dataset_name / evaluation_dir_name
            if eval_dir.is_dir():
                for eval_file in eval_dir.glob("predictions_*.parquet"):
                    if model_filter in eval_file.name and re.match(window_regex, eval_file.name):
                        df = pd.read_parquet(eval_file)
                        model_match = re.match(window_regex, eval_file.name)
                        model_name = model_match.group(1)
                        window_index = int(model_match.group(2))
                        df = prepare_predictions_df(df, actuals_df=pd.DataFrame())
                        df_melted = df.melt(
                            id_vars=["unique_id", "ds", "horizon", "y"],
                            var_name="model",
                            value_name="y_pred"
                        )
                        df_melted["tso"] = normalized_tso_name.replace(" ", "_")
                        df_melted["window_index"] = window_index
                        all_dfs.append(df_melted)
    concatenated = pd.concat(all_dfs, ignore_index=True)
    return concatenated.pivot_table(
        index=["tso", "unique_id", "ds", "horizon", "y", "window_index"],
        columns="model",
        values="y_pred"
    ).reset_index()