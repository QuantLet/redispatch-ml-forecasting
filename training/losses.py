from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from neuralforecast.losses.pytorch import MQLoss, MAE, BasePointLoss


# ── Helpers ────────────────────────────────────────────────────────────────────

def _divide_no_nan(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise division, replacing NaN / ±Inf with 0."""
    return torch.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_mean(losses: Tensor, weights: Tensor) -> Tensor:
    """Weighted mean over per-horizon losses."""
    return _divide_no_nan(torch.sum(losses * weights), torch.sum(weights))


def compute_weights(
    horizon_weight: Tensor | None,
    y: Tensor,
    mask: Tensor | None,
) -> Tensor:
    """Build final per-datapoint weights from ``horizon_weight`` and ``mask``."""
    if mask is None:
        mask = torch.ones_like(y, device=y.device)
    if horizon_weight is None:
        horizon_weight = torch.ones(mask.shape[-1])
    else:
        assert mask.shape[-1] == len(horizon_weight), (
            "horizon_weight must have same length as Y"
        )
    weights = torch.ones_like(mask, device=mask.device) * horizon_weight.to(mask.device)
    return weights * mask


# ── MQMedianLoss (existing) ───────────────────────────────────────────────────

class MQMedianLoss(MQLoss):
    """Point forecast loss which uses the median and MAE."""
    def __init__(self, level: list[float], horizon_weight: list[float] | None = None):
        super().__init__(level=level, horizon_weight=horizon_weight)
        self.median_loc = len(level)
        self.mae_loss = MAE(horizon_weight=horizon_weight)

    def __call__(
        self,
        y: Tensor,
        y_hat: Tensor,
        mask: Tensor | None = None,
        y_insample: Tensor | None = None,
    ):
        return self.mae_loss(y=y, y_hat=y_hat[:, :, :, self.median_loc], mask=mask, y_insample=y_insample)


# ── Hurdle / zero-inflated losses ──────────────────────────────────────────────

class HurdleCouplingPointLoss(BasePointLoss):
    """
    Hurdle loss with a 3-stage learning schedule.

    Outputs ``[B, H, 2]``:
      - channel 0 → logit for P(non-zero)
      - channel 1 → magnitude prediction

    Components:
      - **BCE** on unscaled zero / non-zero classification.
      - **Huber** on magnitude (only non-zero samples), with per-direction δ.
      - **Coupling** term ``|p · ReLU(m̂) − y|`` (Huber).
    """

    outputsize_multiplier = 2

    def __init__(
        self,
        horizon_weight: np.ndarray | None = None,
        eps: float = 1e-6,
        pos_weight: float | dict | None = None,
        static_df: pd.DataFrame | None = None,
        delta_map: dict[str, float] | None = None,
        schedule_steps: tuple[int, int] = (500, 1000),
        lambda_bce_final: float = 0.25,
    ):
        super().__init__(
            horizon_weight=horizon_weight,
            outputsize_multiplier=2,
            output_names=["non_zero_logit", "magnitude"],
        )
        self.eps = eps
        self._static_df = static_df
        self.lambda_bce_final = float(lambda_bce_final)
        self.schedule_steps = tuple(int(x) for x in schedule_steps)
        self._register_pos_weights(pos_weight)
        self._register_deltas(delta_map)
        self.mae_loss = nn.L1Loss(reduction="none")
        self._last_components: dict[str, float] = {}
        self._last_weights: dict[str, float] = {}

    # ── buffer registration ────────────────────────────────────────────────
    def _register_deltas(self, delta_map: dict[str, float] | None):
        if delta_map is None or self._static_df is None:
            raise ValueError("delta_map and static_df are required for per-direction Huber loss.")
        static_cols = self._static_df.columns.difference(["unique_id"])
        delta_vals = np.zeros(len(static_cols), dtype=np.float32)
        for uid in sorted(self._static_df["unique_id"]):
            idx = (
                self._static_df
                .loc[self._static_df["unique_id"] == uid, static_cols]
                .to_numpy()
                .argmax()
            )
            delta_vals[idx] = float(delta_map[uid])
        self.register_buffer("delta_vec", torch.tensor(delta_vals, dtype=torch.float32), persistent=True)

    def _register_pos_weights(self, pos_weight: float | dict | None):
        if pos_weight is not None:
            if isinstance(pos_weight, dict):
                if self._static_df is None:
                    raise ValueError("static_df must be provided when pos_weight is a dict.")
                static_cols = self._static_df.columns.difference(["unique_id"])
                pw_values = np.zeros(len(static_cols), dtype=np.float32)
                for key in sorted(pos_weight.keys()):
                    idx = (
                        self._static_df
                        .loc[self._static_df["unique_id"] == key, static_cols]
                        .to_numpy()
                        .argmax()
                    )
                    pw_values[idx] = pos_weight[key]
                pw = torch.tensor(pw_values, dtype=torch.float32)
                self._pos_weight_dict = pos_weight
            else:
                pw = torch.as_tensor(pos_weight, dtype=torch.float32)
                if pw.ndim == 0:
                    pw = pw.reshape(1)
                self._pos_weight_dict = None
            self.register_buffer("pos_weight", pw, persistent=True)
        else:
            self.pos_weight = None
            self._pos_weight_dict = None

    # ── Huber helper ───────────────────────────────────────────────────────
    @staticmethod
    def _huber(diff: Tensor, delta: Tensor) -> Tensor:
        abs_diff = diff.abs()
        return torch.where(
            abs_diff <= delta,
            0.5 * abs_diff * abs_diff / (delta + 1e-12),
            abs_diff - 0.5 * delta,
        )

    def _per_sample_vec(self, vec: Tensor, stat_exog: Tensor) -> Tensor:
        """``stat_exog`` [B, D] · ``vec`` [D] → [B, 1]."""
        return (stat_exog.to(vec.dtype) * vec[None, :]).sum(dim=-1, keepdim=True)

    # ── schedule ───────────────────────────────────────────────────────────
    def _component_weights(self, step: int) -> dict[str, float]:
        a, b = self.schedule_steps
        if step <= a:
            wbce, wmag, wcoup = 1.0, 0.0, 0.0
        elif step <= b:
            r = (step - a) / max(b - a, 1)
            wbce = 1.0 - (1.0 - self.lambda_bce_final) * r
            wmag = r
            wcoup = r * r
        else:
            wbce, wmag, wcoup = self.lambda_bce_final, 1.0, 1.0
        return {"bce": wbce, "mag": wmag, "coup": wcoup}

    # ── forward ────────────────────────────────────────────────────────────
    def __call__(
        self,
        y: Tensor,
        y_unscaled: Tensor,
        y_hat: Tensor,
        y_insample: Tensor | None = None,
        mask: Tensor | None = None,
        stat_exog: Tensor | None = None,
        step: int | None = None,
    ) -> Tensor:
        # Squeeze extra dims produced by the framework
        if y.dim() == 3:
            y = y.squeeze(-1)
        if y_hat.dim() == 4:
            y_hat = y_hat.squeeze(2)
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(-1)

        pos_logit = y_hat[..., 0]       # [B, H]
        magnitude_hat = y_hat[..., 1]   # [B, H]

        valid_mask = (
            torch.ones_like(pos_logit, dtype=pos_logit.dtype)
            if mask is None else mask.to(pos_logit.dtype)
        )
        non_zero = (y_unscaled >= self.eps).to(pos_logit.dtype)

        # ── BCE on classification head ─────────────────────────────────
        pw = None
        if self.pos_weight is not None:
            pw_vec = self.pos_weight.to(pos_logit.device)
            if pw_vec.numel() > 1 and stat_exog is not None:
                pw = self._per_sample_vec(pw_vec, stat_exog)
            else:
                pw = pw_vec
        clf_losses = F.binary_cross_entropy_with_logits(
            pos_logit, non_zero, reduction="none", pos_weight=pw,
        )
        denom_clf = valid_mask.sum(dim=0)
        clf_loss_h = (clf_losses * valid_mask).sum(dim=0) / (denom_clf + 1e-8)

        # ── Huber on magnitude (non-zero only) ────────────────────────
        if stat_exog is None:
            raise ValueError("stat_exog is required for per-direction Huber δ selection.")
        delta = self._per_sample_vec(self.delta_vec.to(y.device), stat_exog)  # [B, 1]

        diff = magnitude_hat - y
        mag_losses = self._huber(diff, delta)
        reg_mask = valid_mask * non_zero
        denom_reg = reg_mask.sum(dim=0)
        mag_loss_h = (mag_losses * reg_mask).sum(dim=0) / (denom_reg + 1e-8)

        # ── Coupling term ──────────────────────────────────────────────
        pos_prob = torch.sigmoid(pos_logit)
        combined_pred = pos_prob * F.relu(magnitude_hat)
        coupling_losses = self._huber(combined_pred - y, delta)
        coupling_loss_h = (coupling_losses * valid_mask).sum(dim=0) / (denom_clf + 1e-8)

        # ── Schedule-weighted sum ──────────────────────────────────────
        ws = self._component_weights(int(step) if step is not None else self.schedule_steps[-1])
        total_loss_h = (
            ws["bce"] * clf_loss_h
            + ws["mag"] * mag_loss_h
            + ws["coup"] * coupling_loss_h
        )

        weights = compute_weights(horizon_weight=self.horizon_weight, y=y, mask=mask)

        self._last_components = {
            "bce_loss": float(weighted_mean(clf_loss_h, weights).detach().cpu()),
            "mag_loss": float(weighted_mean(mag_loss_h, weights).detach().cpu()),
            "coupling_loss": float(weighted_mean(coupling_loss_h, weights).detach().cpu()),
        }
        self._last_weights = ws
        return weighted_mean(total_loss_h, weights)

    def get_last_components(self) -> dict[str, float]:
        return getattr(self, "_last_components", {})

    def get_task_weights(self) -> dict[str, float]:
        return getattr(self, "_last_weights", {})


class HurdleEvalLoss(BasePointLoss):
    """
    Validation / eval loss for hurdle models.

    Combines the two output channels into an expected-value prediction
    ``p̂ · m̂`` and scores it with MSE or MAE against the raw target.
    """

    def __init__(
        self,
        horizon_weight: np.ndarray | None = None,
        loss_name: str = "mae",
    ):
        super().__init__(horizon_weight=horizon_weight, outputsize_multiplier=1)
        if loss_name.lower() == "mse":
            self.final_loss = nn.MSELoss(reduction="none")
        elif loss_name.lower() == "mae":
            self.final_loss = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

    def __call__(
        self,
        y: Tensor,
        y_hat: Tensor,
        y_insample: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if y.dim() == 3:
            y = y.squeeze(-1)
        if y_hat.dim() == 4:
            y_hat = y_hat.squeeze(2)
        if mask is not None and mask.dim() == 3:
            mask = mask.squeeze(-1)

        pos_logit = y_hat[..., 0]
        magnitude_hat = y_hat[..., 1]

        valid_mask = (
            torch.ones_like(pos_logit, dtype=pos_logit.dtype)
            if mask is None else mask.to(pos_logit.dtype)
        )

        expected_pred = torch.sigmoid(pos_logit) * magnitude_hat  # [B, H]
        losses_h = self.final_loss(expected_pred, y)               # [B, H]
        denom = valid_mask.sum(dim=0)
        loss_per_h = (losses_h * valid_mask).sum(dim=0) / (denom + 1e-8)

        weights = compute_weights(horizon_weight=self.horizon_weight, y=y, mask=mask)
        return weighted_mean(loss_per_h, weights)