import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_col_dict import AISColDict
from ModelUtils.loss_calculator import LossCalculator, LossTypes


@dataclass
class Config:
    device: str


class Model(nn.Module):
    def __init__(self, loss_calculator: LossCalculator, cfg: Config):
        super().__init__()
        self._cfg = cfg
        self.loss_calculator = loss_calculator

        # Number of AIS features
        self._num_ais_features = len(AISColDict)

    def __str__(self):
        return "lerp"

    @staticmethod
    def _compute_deltas_meters(lat1, lon1, lat2, lon2):
        """
        Computes north/east displacement in meters using equirectangular approximation.
        """
        R = 6371000.0  # Earth radius in meters
        lat1_rad = lat1 * math.pi / 180
        lat2_rad = lat2 * math.pi / 180
        dlat = (lat2 - lat1) * math.pi / 180
        dlon = (lon2 - lon1) * math.pi / 180

        delta_north = dlat * R
        delta_east = dlon * R * torch.cos((lat1_rad + lat2_rad) / 2)
        return delta_north, delta_east

    def forward(self, ais_batch: AISBatch) -> tuple[LossTypes, tuple[torch.Tensor, ...]]:
        # Ground truth
        true_lats = ais_batch.lats.to(self._cfg.device)
        true_lons = ais_batch.lons.to(self._cfg.device)

        # Start with GT and interpolate missing
        lats = true_lats.clone()
        lons = true_lons.clone()

        timestamps = ais_batch.observed_timestamps.to(self._cfg.device)
        masks = ais_batch.masks.to(self._cfg.device)

        # Position valid if BOTH deltas are observed
        pos_valid_mask = (
            masks[:, :, AISColDict.NORTHERN_DELTA.value]
            * masks[:, :, AISColDict.EASTERN_DELTA.value]
        ).bool()

        B, S = lats.shape

        # Linear interpolation
        for b in range(B):
            valid_idx = torch.where(pos_valid_mask[b])[0]
            if valid_idx.numel() < 2:
                continue
            for i in range(valid_idx.numel() - 1):
                t0 = valid_idx[i]
                t1 = valid_idx[i + 1]
                if t1 <= t0 + 1:
                    continue
                time0, time1 = timestamps[b, t0], timestamps[b, t1]
                denom = time1 - time0
                if denom == 0:
                    continue
                lat0, lon0 = lats[b, t0], lons[b, t0]
                lat1, lon1 = lats[b, t1], lons[b, t1]
                for t in range(t0 + 1, t1):
                    alpha = (timestamps[b, t] - time0) / denom
                    lats[b, t] = (1 - alpha) * lat0 + alpha * lat1
                    lons[b, t] = (1 - alpha) * lon0 + alpha * lon1

        # Full trajectory tensors
        full_traj_pred = torch.stack([lats, lons], dim=-1)
        full_traj_true = torch.stack([true_lats, true_lons], dim=-1)

        # --- Only masked positions ---
        missing_mask = ~pos_valid_mask
        pos_pred = full_traj_pred[missing_mask]
        pos_true = full_traj_true[missing_mask]

        if pos_pred.numel() == 0:
            pos_pred = torch.zeros(B, 1, 2, device=self._cfg.device)
            pos_true = torch.zeros_like(pos_pred)
        else:
            pos_pred = pos_pred.view(B, -1, 2)
            pos_true = pos_true.view(B, -1, 2)

        # --- Deltas only at missing positions ---
        num_missing = pos_pred.shape[1]
        deltas_pred = torch.zeros(B, num_missing, self._num_ais_features, device=self._cfg.device)
        deltas_true = torch.zeros_like(deltas_pred)

        # Fill deltas for missing positions only
        for b in range(B):
            missing_idx = torch.where(missing_mask[b])[0]
            for i, t in enumerate(missing_idx):
                if t == 0:
                    continue  # can't compute delta for first timestep
                # Predicted
                dn, de = self._compute_deltas_meters(lats[b, t - 1], lons[b, t - 1], lats[b, t], lons[b, t])
                deltas_pred[b, i, AISColDict.NORTHERN_DELTA.value] = dn
                deltas_pred[b, i, AISColDict.EASTERN_DELTA.value] = de
                # True
                dn_true, de_true = self._compute_deltas_meters(
                    true_lats[b, t - 1], true_lons[b, t - 1], true_lats[b, t], true_lons[b, t]
                )
                deltas_true[b, i, AISColDict.NORTHERN_DELTA.value] = dn_true
                deltas_true[b, i, AISColDict.EASTERN_DELTA.value] = de_true

        # Dummy forces
        forces_pred = torch.zeros(B, num_missing, 1, device=self._cfg.device)
        forces_true = torch.zeros_like(forces_pred)
        consistency_loss = torch.tensor(0.0, device=self._cfg.device)

        # --- Compute loss ---
        loss = self.loss_calculator.calculate_loss(
            self.training,
            full_traj_pred,
            full_traj_true,
            pos_pred,
            pos_true,
            deltas_pred,
            deltas_true,
            consistency_loss,
            forces_pred,
            forces_true,
        )

        return loss, (lats, lons, true_lats, true_lons)
