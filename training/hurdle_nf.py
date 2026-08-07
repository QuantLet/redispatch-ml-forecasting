"""
NeuralForecast wrapper for hurdle (zero-inflated) models.

``RedispatchNeuralForecast`` extends the standard ``NeuralForecast`` wrapper to:

1. Apply **local standard scaling** and propagate the per-series scaler stats
   (``μ``, ``σ``) to each hurdle model so they can unscale inside ``forward``.
2. Override ``save`` / ``load`` so that the custom model classes
   (``HurdleNBEATSx``, ``HurdleNHITS``, ``HurdleTFT``) are properly
   (de-)serialised.
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional

import fsspec
import numpy as np
import torch

from coreforecast.grouped_array import GroupedArray
from neuralforecast.core import (
    MODEL_FILENAME_DICT,
    NeuralForecast,
    _type2scaler,
    TimeSeriesDataset,
    _FilesDataset,
)

from training.hurdle_models import HurdleNBEATSx, HurdleNHITS, HurdleTFT

logger = logging.getLogger(__name__)


# Registry: maps lowercase class name → class for load()
_HURDLE_MODEL_REGISTRY: dict[str, type] = {
    "hurdlenbeatsx": HurdleNBEATSx,
    "hurdlenhits": HurdleNHITS,
    "hurdletft": HurdleTFT,
}

MODEL_FILENAME_DICT_REDISPATCH = {**MODEL_FILENAME_DICT, **_HURDLE_MODEL_REGISTRY}


class RedispatchNeuralForecast(NeuralForecast):
    """
    ``NeuralForecast`` subclass that transfers local scaler stats to each
    model and correctly saves / loads hurdle model checkpoints.
    """

    SUPPORTED_LOCAL_SCALERS: list[str] = ["standard"]

    # ── fit-time scaling ───────────────────────────────────────────────────
    def _scalers_fit_transform(self, dataset: TimeSeriesDataset) -> None:
        self.scalers_ = {}
        if self.local_scaler_type is None:
            for model in self.models:
                model._local_scalers = None
            return

        if self.local_scaler_type not in self.SUPPORTED_LOCAL_SCALERS:
            raise ValueError(
                f"Only {self.SUPPORTED_LOCAL_SCALERS} scaler types are supported."
            )

        for i, col in enumerate(dataset.temporal_cols):
            if col == "available_mask":
                continue
            ga = GroupedArray(dataset.temporal[:, i].numpy(), dataset.indptr)
            self.scalers_[col] = _type2scaler[self.local_scaler_type]().fit(ga)
            dataset.temporal[:, i] = torch.from_numpy(
                self.scalers_[col].transform(ga)
            )

        # Propagate scaler μ/σ to every model that understands it.
        for model in self.models:
            y_col_name = dataset.temporal_cols[dataset.y_idx]
            model._y_local_scaler_stats = self.scalers_[y_col_name].stats_

    def _scalers_target_inverse_transform(
        self, data: np.ndarray, indptr: np.ndarray,
    ) -> np.ndarray:
        # Hurdle models unscale internally → no-op here.
        return data

    # ── save ───────────────────────────────────────────────────────────────
    def save(
        self,
        path: str,
        model_index: list[int] | None = None,
        save_dataset: bool = True,
        overwrite: bool = False,
    ):
        if path.endswith("/"):
            path = path[:-1]
        if model_index is None:
            model_index = list(range(len(self.models)))

        fs, _, _ = fsspec.get_fs_token_paths(path)
        if not fs.exists(path):
            fs.makedirs(path)
        else:
            files = fs.ls(path)
            if files:
                if not overwrite:
                    raise Exception(
                        "Directory is not empty. Set overwrite=True to overwrite files."
                    )
                fs.rm(path, recursive=True)
                fs.mkdir(path)

        count_names: dict[str, int] = {"model": 0}
        alias_to_model: dict[str, str] = {}
        for i, model in enumerate(self.models):
            if i not in model_index:
                continue
            model_name = repr(model)
            model_class_name = model.__class__.__name__.lower()
            if model_class_name not in MODEL_FILENAME_DICT_REDISPATCH:
                base_name = model.__class__.__base__.__name__.lower()
                if base_name in MODEL_FILENAME_DICT_REDISPATCH:
                    model_class_name = base_name
                else:
                    raise ValueError(
                        f"Model {model.__class__.__name__} is not supported for saving."
                    )
            alias_to_model[model_name] = model_class_name
            count_names[model_name] = count_names.get(model_name, -1) + 1
            model.save(f"{path}/{model_name}_{count_names[model_name]}.ckpt")

        with fsspec.open(f"{path}/alias_to_model.pkl", "wb") as f:
            pickle.dump(alias_to_model, f)

        if save_dataset and hasattr(self, "dataset"):
            if isinstance(self.dataset, _FilesDataset):
                raise ValueError(
                    "Cannot save distributed dataset. "
                    "Set save_dataset=False and pass df at predict-time."
                )
            with fsspec.open(f"{path}/dataset.pkl", "wb") as f:
                pickle.dump(self.dataset, f)
        elif save_dataset:
            raise Exception(
                "You need to have a stored dataset to save it; "
                "set save_dataset=False to skip."
            )

        config_dict = {
            "h": self.h,
            "freq": self.freq,
            "_fitted": self._fitted,
            "local_scaler_type": self.local_scaler_type,
            "scalers_": self.scalers_,
            "id_col": self.id_col,
            "time_col": self.time_col,
            "target_col": self.target_col,
        }
        for attr in ["prediction_intervals", "_cs_df"]:
            config_dict[attr] = getattr(self, attr, None)
        if save_dataset:
            config_dict.update({
                "uids": self.uids,
                "last_dates": self.last_dates,
                "ds": self.ds,
            })
        with fsspec.open(f"{path}/configuration.pkl", "wb") as f:
            pickle.dump(config_dict, f)

    # ── load ───────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str, verbose: bool = False, **kwargs):
        if path.endswith("/"):
            path = path[:-1]

        fs, _, _ = fsspec.get_fs_token_paths(path)
        files = [f.split("/")[-1] for f in fs.ls(path) if fs.isfile(f)]
        models_ckpt = [f for f in files if f.endswith(".ckpt")]
        if not models_ckpt:
            raise Exception("No model found in directory.")

        if verbose:
            print(10 * "-" + " Loading models " + 10 * "-")

        models = []
        try:
            with fsspec.open(f"{path}/alias_to_model.pkl", "rb") as f:
                alias_to_model = pickle.load(f)
        except FileNotFoundError:
            alias_to_model = {}

        for model_file in models_ckpt:
            model_name = "_".join(model_file.split("_")[:-1])
            model_class_name = alias_to_model.get(model_name, model_name)
            if verbose:
                print(f"Loading {model_name} ({model_class_name})")
            loaded = MODEL_FILENAME_DICT_REDISPATCH[model_class_name].load(
                f"{path}/{model_file}", **kwargs,
            )
            loaded.alias = model_name
            models.append(loaded)
            if verbose:
                print(f"  → {model_name} loaded.")

        if verbose:
            print(10 * "-" + " Loading dataset " + 10 * "-")
        try:
            with fsspec.open(f"{path}/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            if verbose:
                print("Dataset loaded.")
        except FileNotFoundError:
            dataset = None
            if verbose:
                print("No dataset found in directory.")

        if verbose:
            print(10 * "-" + " Loading configuration " + 10 * "-")
        try:
            with fsspec.open(f"{path}/configuration.pkl", "rb") as f:
                config_dict = pickle.load(f)
            if verbose:
                print("Configuration loaded.")
        except FileNotFoundError:
            raise Exception("No configuration found in directory.")

        default_scalar_type = getattr(dataset, "local_scaler_type", None)
        default_scalars_ = getattr(dataset, "scalers_", None)

        nf = cls(
            models=models,
            freq=config_dict["freq"],
            local_scaler_type=config_dict.get("local_scaler_type", default_scalar_type),
        )
        for attr, default in [("id_col", "unique_id"), ("time_col", "ds"), ("target_col", "y")]:
            setattr(nf, attr, config_dict.get(attr, default))
        for attr in ["prediction_intervals", "_cs_df"]:
            setattr(nf, attr, config_dict.get(attr, None))

        if dataset is not None:
            nf.dataset = dataset
            for attr in ["uids", "last_dates", "ds"]:
                setattr(nf, attr, config_dict[attr])

        nf._fitted = config_dict["_fitted"]
        nf.scalers_ = config_dict.get("scalers_", default_scalars_)
        return nf
