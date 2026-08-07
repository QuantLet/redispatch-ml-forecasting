"""
Hurdle (zero-inflated) wrappers for NBEATSx, NHITS, and TFT.

Each model produces a ``[B, H, 2]`` output:
  - channel 0 → logit for P(non-zero)  ("zero_probability")
  - channel 1 → magnitude prediction   ("magnitude")

The wrappers override ``forward``, ``training_step``, ``validation_step``,
``_predict_step_direct_batch``, ``fit``, and serialisation hooks so that the
hurdle loss receives **both** scaled and unscaled targets.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralforecast.losses.pytorch import MAE, BasePointLoss
from neuralforecast.models import NBEATSx, TFT
from neuralforecast.models.nhits import NHITS
from neuralforecast.tsdataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared mixin - common hurdle logic for *every* architecture
# ═══════════════════════════════════════════════════════════════════════════════

class _HurdleMixin(nn.Module):
    """
    Mixin injected **before** the concrete Nixtla model class so that its
    methods take precedence (Python MRO).

    Requires the concrete class to define:
      * ``self.loss``  with ``outputsize_multiplier == 2``
      * ``self.blocks`` (for NBEATSx / NHITS) *or* the TFT body
    """

    # Parameters that are hurdle-specific and must be serialised separately.
    CUSTOM_PARAMS = {
        "per_series_base_rates",
        "per_series_median_z",
        "scale_magnitude_head",
    }

    # ── initialisation helpers ─────────────────────────────────────────────
    def _hurdle_init(
        self,
        per_series_base_rates: dict[str, float],
        per_series_median_z: dict[str, float],
        scale_magnitude_head: bool = True,
    ):
        self.base_rates = per_series_base_rates
        self.median_z = per_series_median_z
        self.scale_magnitude_head = scale_magnitude_head
        self._initialized_static_features = False
        self.save_hyperparameters(
            *list(self.CUSTOM_PARAMS),
            ignore=["loss", "valid_loss"],
        )

    # ── static-direction feature buffers ───────────────────────────────────
    def _initialize_static_dir_features(
        self,
        n_groups: int,
        static_cols: pd.Index,
        static: "np.ndarray | torch.Tensor",
    ):
        def logit(p, eps=1e-3):
            p = np.clip(p, eps, 1 - eps)
            return float(np.log(p / (1 - p)))

        local_scaler_stats = getattr(self, "_y_local_scaler_stats", None)
        if local_scaler_stats is None:
            raise ValueError(
                "Call fit() after local scaler stats are computed before "
                "initializing static direction features."
            )

        original_unique_ids = sorted(self.base_rates.keys())
        if len(static_cols) > n_groups:
            raise NotImplementedError(
                "Static exogenous features other than direction indicators "
                "are not supported in hurdle models."
            )

        y_mu_arr = np.zeros(len(static_cols))
        y_sig_arr = np.zeros(len(static_cols))
        mu_bias = np.zeros(len(static_cols))
        logit_base_rates = np.zeros(len(static_cols))

        for nixtla_idx, (original_id, static_row) in enumerate(
            zip(original_unique_ids, static)
        ):
            id_direction_index = torch.argmax(static_row)
            id_base_rate = self.base_rates[original_id]
            y_mu_arr[id_direction_index] = local_scaler_stats[nixtla_idx, 0]
            y_sig_arr[id_direction_index] = local_scaler_stats[nixtla_idx, 1]
            logit_base_rates[id_direction_index] = logit(id_base_rate)

            if isinstance(self.median_z, dict):
                if original_id not in self.median_z:
                    raise ValueError(
                        "Median per direction must be provided for all series."
                    )
                median_direction = self.median_z[original_id]
            else:
                median_direction = float(self.median_z)

            if self.scale_magnitude_head:
                mu_bias[id_direction_index] = (
                    (median_direction - y_mu_arr[id_direction_index])
                    / (y_sig_arr[id_direction_index] + 1e-8)
                )
            else:
                mu_bias[id_direction_index] = median_direction

        if not self._initialized_static_features:
            self.register_buffer(
                "dir_logit_bias",
                torch.tensor(logit_base_rates, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "y_mu",
                torch.tensor(y_mu_arr, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "y_sig",
                torch.tensor(y_sig_arr, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "dir_mu_bias",
                torch.tensor(mu_bias, dtype=torch.float32),
                persistent=True,
            )
            self._initialized_static_features = True

    # ── unscaling helpers ──────────────────────────────────────────────────
    def _unscale(self, stat_exog: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse z-score: ``y = y_scaled * σ + μ``."""
        dir_oh = stat_exog.to(y_scaled.dtype)
        if y_scaled.dim() == 3:
            y_scaled = y_scaled.squeeze(-1)
        y_mu = (dir_oh * self.y_mu[None, :]).sum(dim=-1)
        y_sig = (dir_oh * self.y_sig[None, :]).sum(dim=-1)
        return y_scaled * y_sig[:, None] + y_mu[:, None]

    def _get_min_eps(self, reference: torch.Tensor) -> torch.Tensor:
        eps = getattr(self.loss, "eps", torch.tensor(1e-1))
        if not isinstance(eps, torch.Tensor):
            eps = torch.tensor(eps, device=reference.device, dtype=reference.dtype)
        return eps.to(device=reference.device, dtype=reference.dtype)

    # ── logging helper ─────────────────────────────────────────────────────
    def _log_lr(self):
        opt = self.optimizers()
        if opt is not None:
            pg = (
                opt[0].param_groups if isinstance(opt, (list, tuple)) else opt.param_groups
            )
            lr_val = pg[0].get("lr") if pg else None
            if lr_val is not None:
                self.log("lr", lr_val, on_step=True, prog_bar=False)

    # ── training_step (replaces base class) ────────────────────────────────
    def training_step(self, batch, batch_idx):
        if self.RECURRENT:
            self.h = self.h_train

        y_idx = batch["y_idx"]
        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(
            batch, step="train",
        )
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="train",
        )
        # Capture *un-normalised* outsample_y before the scaler touches it
        original_outsample_y = windows["temporal"][:, self.input_size :, y_idx]

        windows = self._normalization(windows=windows, y_idx=y_idx)
        (
            insample_y, insample_mask,
            outsample_y, outsample_mask,
            hist_exog, futr_exog, stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        output = self(windows_batch)
        output = self.loss.domain_map(output)

        unscaled_outsample_y = self._unscale(stat_exog=stat_exog, y_scaled=original_outsample_y)
        processed_y = original_outsample_y if self.scale_magnitude_head else unscaled_outsample_y

        loss = self.loss(
            y=processed_y,
            y_unscaled=unscaled_outsample_y,
            y_hat=output,
            y_insample=insample_y,
            mask=outsample_mask,
            stat_exog=stat_exog,
            step=self.global_step,
        )

        if torch.isnan(loss):
            raise Exception("Loss is NaN, training stopped.")

        train_loss_log = loss.detach().item()
        self.log("train_loss", train_loss_log, batch_size=outsample_y.size(0), prog_bar=True, on_epoch=True)
        self.train_trajectories.append((self.global_step, train_loss_log))

        # ── diagnostics ───────────────────────────────────────────────
        non_zero = (unscaled_outsample_y >= self._get_min_eps(unscaled_outsample_y)).detach().cpu().numpy()
        self.log("non_zero_pct", non_zero.sum() / np.prod(non_zero.shape) * 100,
                 batch_size=outsample_y.size(0), on_epoch=True)
        self._log_lr()
        if hasattr(self.loss, "get_task_weights"):
            for k, v in self.loss.get_task_weights().items():
                self.log(k, v, batch_size=outsample_y.size(0), on_epoch=True)
        if hasattr(self.loss, "get_last_components"):
            for k, v in self.loss.get_last_components().items():
                self.log(k, v, batch_size=outsample_y.size(0), on_epoch=True)

        with torch.no_grad():
            eps = self._get_min_eps(unscaled_outsample_y)
            p_hat = torch.sigmoid(output[..., 0])
            self.log("mean_p_hat", p_hat.mean().item(), batch_size=outsample_y.size(0), on_epoch=True)

            is_zero = (unscaled_outsample_y < eps).float()
            n_zeros = is_zero.sum()
            if n_zeros > 0:
                relu_mag = F.relu(output[..., 1])
                self.log("mean_relu_mag_on_zeros",
                         (relu_mag * is_zero).sum().item() / n_zeros.item(),
                         batch_size=outsample_y.size(0), on_epoch=True)

            is_pos = (unscaled_outsample_y >= eps).float()
            n_pos = is_pos.sum()
            if n_pos > 0:
                mae_on_pos = (torch.abs(output[..., 1] - processed_y.squeeze(-1)) * is_pos).sum() / n_pos
                self.log("mae_mag_on_positives", mae_on_pos.item(), batch_size=outsample_y.size(0), on_epoch=True)

        self.h = self.horizon_backup
        return loss

    # ── validation_step ────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(batch, step="val")
        n_windows = len(windows_temporal)
        y_idx = batch["y_idx"]

        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
            w_idxs = np.arange(
                i * windows_batch_size,
                min((i + 1) * windows_batch_size, n_windows),
            )
            windows = self._sample_windows(
                windows_temporal, static, static_cols, temporal_cols,
                step="val", w_idxs=w_idxs,
            )
            original_outsample_y = torch.clone(
                windows["temporal"][:, self.input_size :, y_idx]
            )
            windows = self._normalization(windows=windows, y_idx=y_idx)
            (
                insample_y, insample_mask, _, outsample_mask,
                hist_exog, futr_exog, stat_exog,
            ) = self._parse_windows(batch, windows)

            if self.RECURRENT:
                output_batch = self._validate_step_recurrent_batch(
                    insample_y=insample_y, insample_mask=insample_mask,
                    futr_exog=futr_exog, hist_exog=hist_exog,
                    stat_exog=stat_exog, y_idx=y_idx,
                )
            else:
                windows_batch = dict(
                    insample_y=insample_y, insample_mask=insample_mask,
                    futr_exog=futr_exog, hist_exog=hist_exog,
                    stat_exog=stat_exog,
                )
                output_batch = self(windows_batch)

            output_batch = self.loss.domain_map(output_batch)
            valid_loss_batch = self._compute_valid_loss(
                insample_y=insample_y,
                outsample_y=original_outsample_y,
                output=output_batch,
                outsample_mask=outsample_mask,
                stat_exog=stat_exog,
                y_idx=batch["y_idx"],
            )
            valid_losses.append(valid_loss_batch)
            batch_sizes.append(len(output_batch))

        valid_loss = torch.stack(valid_losses)
        batch_sizes_t = torch.tensor(batch_sizes, device=valid_loss.device)
        batch_size = torch.sum(batch_sizes_t)
        valid_loss = torch.sum(valid_loss * batch_sizes_t) / batch_size

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        valid_loss_log = valid_loss.detach()
        self.log("valid_loss", valid_loss_log.item(), batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.append(valid_loss_log)
        return valid_loss

    # ── valid-loss helper ──────────────────────────────────────────────────
    def _compute_valid_loss(
        self, insample_y, outsample_y, output, outsample_mask, stat_exog, y_idx
    ):
        output = self._inv_normalization(y_hat=output, y_idx=y_idx)
        if not self.scale_magnitude_head:
            outsample_y = self._unscale(stat_exog=stat_exog, y_scaled=outsample_y)
        if outsample_mask.dim() == 3:
            outsample_mask = outsample_mask.squeeze(-1)
        if outsample_y.dim() == 3:
            outsample_y = outsample_y.squeeze(-1)
        return self.valid_loss(
            y_insample=insample_y,
            mask=outsample_mask,
            y=outsample_y,
            y_hat=output,
        )

    # ── predict helper ─────────────────────────────────────────────────────
    def _predict_step_direct_batch(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx,
    ):
        windows_batch = dict(
            insample_y=insample_y, insample_mask=insample_mask,
            futr_exog=futr_exog, hist_exog=hist_exog, stat_exog=stat_exog,
        )
        output_batch = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch)

        if self.scale_magnitude_head:
            mu_hat = self._unscale(stat_exog=stat_exog, y_scaled=output_batch[:, :, 1])
        else:
            mu_hat = output_batch[:, :, 1]

        y_hat = torch.stack([output_batch[:, :, 0], mu_hat], dim=-1)
        return y_hat

    # ── fit (injects static features) ──────────────────────────────────────
    def fit(
        self,
        dataset: TimeSeriesDataset,
        val_size: int = 0,
        test_size: int = 0,
        random_seed: int | None = None,
        distributed_config=None,
    ):
        if dataset.static is None or dataset.static_cols is None:
            raise ValueError(
                "Static exogenous features (direction indicators) are required "
                "for hurdle models."
            )
        self._initialize_static_dir_features(
            dataset.n_groups, dataset.static_cols, dataset.static,
        )
        return self._fit(
            dataset=dataset,
            batch_size=self.batch_size,
            valid_batch_size=self.valid_batch_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=distributed_config,
        )

    # ── serialisation ──────────────────────────────────────────────────────
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        buffer_keys = ["dir_logit_bias", "y_mu", "y_sig", "dir_mu_bias"]
        has_buffers = all(f"{prefix}{k}" in state_dict for k in buffer_keys)
        if not has_buffers:
            raise ValueError(
                "State dict is missing required hurdle buffers for scaling."
            )
        if has_buffers and not self._initialized_static_features:
            for k in buffer_keys:
                self.register_buffer(k, torch.zeros_like(state_dict[f"{prefix}{k}"]), persistent=True)
            self._initialized_static_features = True
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @classmethod
    def load(cls, path, **kwargs):
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        with fsspec.open(path, "rb") as f, warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            content = torch.load(f, **kwargs)

        from neuralforecast.common._base_model import _disable_torch_init

        with _disable_torch_init():
            hparams = content["hyper_parameters"].copy()
            custom_params = {}
            for p in cls.CUSTOM_PARAMS:
                if p in hparams:
                    custom_params[p] = hparams.pop(p)
            model = cls(**custom_params, **hparams)

        if "assign" in inspect.signature(model.load_state_dict).parameters:
            model.load_state_dict(content["state_dict"], strict=False, assign=True)
        else:
            model.load_state_dict(content["state_dict"], strict=False)
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete hurdle model classes
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# NBEATSx
# ---------------------------------------------------------------------------

class HurdleNBEATSx(_HurdleMixin, NBEATSx):
    """NBEATSx with a two-channel hurdle head (logit + magnitude)."""

    def __init__(
        self,
        h,
        input_size,
        per_series_base_rates: dict[str, float],
        per_series_median_z: dict[str, float],
        scale_magnitude_head: bool = True,
        **kwargs,
    ):
        # Strip our custom params before passing to the parent
        kwargs.pop("per_series_base_rates", None)
        kwargs.pop("per_series_median_z", None)
        kwargs.pop("scale_magnitude_head", None)
        super().__init__(h=h, input_size=input_size, **kwargs)
        self._hurdle_init(per_series_base_rates, per_series_median_z, scale_magnitude_head)

    def on_fit_start(self):
        self._zero_init_block_thetas()
        super(_HurdleMixin, self).on_fit_start()

    def _zero_init_block_thetas(self):
        for b in self.blocks:
            last = b.layers[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, windows_batch):
        if self.decompose_forecast:
            raise NotImplementedError("decompose_forecast not supported for hurdle models")

        insample_y = windows_batch["insample_y"].squeeze(-1)
        insample_mask = windows_batch["insample_mask"].squeeze(-1)
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        B = insample_y.size(0)
        C = self.loss.outputsize_multiplier
        assert C == 2, "HurdleNBEATSx requires outputsize_multiplier == 2"

        dir_oh = stat_exog.to(insample_y.dtype)
        mu0 = (dir_oh * self.dir_mu_bias[None, :]).sum(dim=-1)
        b0 = (dir_oh * self.dir_logit_bias[None, :]).sum(dim=-1)

        forecast = insample_y.new_zeros((B, self.h, C))
        forecast[:, :, 0] = b0.unsqueeze(1).expand(-1, self.h)
        forecast[:, :, 1] = mu0.unsqueeze(1).expand(-1, self.h)

        for block in self.blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        return forecast


# ---------------------------------------------------------------------------
# NHITS
# ---------------------------------------------------------------------------

class HurdleNHITS(_HurdleMixin, NHITS):
    """NHITS with a two-channel hurdle head (logit + magnitude)."""

    def __init__(
        self,
        h,
        input_size,
        per_series_base_rates: dict[str, float],
        per_series_median_z: dict[str, float],
        scale_magnitude_head: bool = True,
        **kwargs,
    ):
        kwargs.pop("per_series_base_rates", None)
        kwargs.pop("per_series_median_z", None)
        kwargs.pop("scale_magnitude_head", None)
        super().__init__(h=h, input_size=input_size, **kwargs)
        self._hurdle_init(per_series_base_rates, per_series_median_z, scale_magnitude_head)

    def on_fit_start(self):
        self._zero_init_block_thetas()
        super(_HurdleMixin, self).on_fit_start()

    def _zero_init_block_thetas(self):
        for b in self.blocks:
            last = b.layers[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, windows_batch):
        if self.decompose_forecast:
            raise NotImplementedError("decompose_forecast not supported for hurdle models")

        insample_y = windows_batch["insample_y"].squeeze(-1).contiguous()
        insample_mask = windows_batch["insample_mask"].squeeze(-1).contiguous()
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        residuals = insample_y.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        B = insample_y.size(0)
        C = self.loss.outputsize_multiplier
        assert C == 2, "HurdleNHITS requires outputsize_multiplier == 2"

        dir_oh = stat_exog.to(insample_y.dtype)
        mu0 = (dir_oh * self.dir_mu_bias[None, :]).sum(dim=-1)
        b0 = (dir_oh * self.dir_logit_bias[None, :]).sum(dim=-1)

        forecast = insample_y.new_zeros((B, self.h, C))
        forecast[:, :, 0] = b0.unsqueeze(1).expand(-1, self.h)
        forecast[:, :, 1] = mu0.unsqueeze(1).expand(-1, self.h)

        for block in self.blocks:
            backcast, block_forecast = block(
                insample_y=residuals,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast

        return forecast


# ---------------------------------------------------------------------------
# TFT
# ---------------------------------------------------------------------------

class HurdleTFT(_HurdleMixin, TFT):
    """TFT with a two-channel hurdle head (logit + magnitude)."""

    def __init__(
        self,
        h,
        input_size,
        per_series_base_rates: dict[str, float],
        per_series_median_z: dict[str, float],
        scale_magnitude_head: bool = True,
        **kwargs,
    ):
        kwargs.pop("per_series_base_rates", None)
        kwargs.pop("per_series_median_z", None)
        kwargs.pop("scale_magnitude_head", None)
        super().__init__(h=h, input_size=input_size, **kwargs)
        self._hurdle_init(per_series_base_rates, per_series_median_z, scale_magnitude_head)

    def forward(self, windows_batch):
        """
        TFT forward.

        The base TFT already produces ``[B, H, outputsize_multiplier]`` via
        ``self.output_adapter``.  We add per-direction baseline biases for
        the logit and magnitude channels exactly as done for NHITS/NBEATSx.
        """
        # --- run the standard TFT body ---
        y_insample = windows_batch["insample_y"]
        futr_exog = windows_batch["futr_exog"]
        hist_exog = windows_batch["hist_exog"]
        stat_exog = windows_batch["stat_exog"]

        if futr_exog is None:
            futr_exog = y_insample[:, [-1]]
            futr_exog = futr_exog.repeat(1, self.example_length, 1)

        s_inp, k_inp, o_inp, t_observed_tgt = self.embedding(
            target_inp=y_insample,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
        )

        if s_inp is not None:
            cs, ce, ch, cc, static_encoder_sparse_weights = self.static_encoder(s_inp)
        else:
            batch_size, example_length, target_size, hidden_size = t_observed_tgt.shape
            cs = torch.zeros(batch_size, hidden_size, device=y_insample.device)
            ce = torch.zeros(batch_size, hidden_size, device=y_insample.device)
            ch = torch.zeros(self.n_rnn_layers, batch_size, hidden_size, device=y_insample.device)
            cc = torch.zeros(self.n_rnn_layers, batch_size, hidden_size, device=y_insample.device)
            static_encoder_sparse_weights = []

        _historical_inputs = [
            k_inp[:, : self.input_size, :],
            t_observed_tgt[:, : self.input_size, :],
        ]
        if o_inp is not None:
            _historical_inputs.insert(0, o_inp[:, : self.input_size, :])
        historical_inputs = torch.cat(_historical_inputs, dim=-2)
        future_inputs = k_inp[:, self.input_size :]

        temporal_features, history_vsn_wgts, future_vsn_wgts = self.temporal_encoder(
            historical_inputs=historical_inputs,
            future_inputs=future_inputs,
            cs=cs, ch=ch, cc=cc,
        )
        temporal_features, attn_wts = self.temporal_fusion_decoder(
            temporal_features=temporal_features, ce=ce,
        )
        self.interpretability_params = {
            "history_vsn_wgts": history_vsn_wgts,
            "future_vsn_wgts": future_vsn_wgts,
            "static_encoder_sparse_weights": static_encoder_sparse_weights,
            "attn_wts": attn_wts,
        }

        y_hat = self.output_adapter(temporal_features)  # [B, H, 2]

        # --- add per-direction baselines ---
        dir_oh = stat_exog.to(y_hat.dtype)
        b0 = (dir_oh * self.dir_logit_bias[None, :]).sum(dim=-1)   # [B]
        mu0 = (dir_oh * self.dir_mu_bias[None, :]).sum(dim=-1)     # [B]

        y_hat[..., 0] = y_hat[..., 0] + b0.unsqueeze(1)
        y_hat[..., 1] = y_hat[..., 1] + mu0.unsqueeze(1)

        return y_hat
