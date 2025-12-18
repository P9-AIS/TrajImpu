import torch
import torch.nn as nn
from dataclasses import dataclass
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.brits import BRITS
from Model.ais_decoder import HeterogeneousAttributeDecoder
from ModelTypes.ais_col_dict import AISColDict
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_stats import AISStats
from ModelUtils.loss_calculator import LossCalculator, LossTypes
from ForceUtils.geo_converter import GeoConverter as GC


@dataclass
class Config:
    device: str

    # encoder
    dim_ais_attr_encoding: int

    # brits
    dim_rnn_hidden: int
    MIT: bool


class Model(nn.Module):
    def __init__(self, dataset_stats: AISStats, loss_calculator: LossCalculator, cfg: Config):

        super().__init__()
        self._cfg = cfg

        self.ais_encoder = HeterogeneousAttributeEncoder(cfg.dim_ais_attr_encoding, dataset_stats).to(cfg.device)

        self.ais_encoding_dim = self.ais_encoder.output_dim

        self.impu_module = BRITS(
            dataset_stats.seq_len, self.ais_encoding_dim,
            cfg.dim_rnn_hidden, MIT=cfg.MIT, device=cfg.device
        ).to(cfg.device)

        self.ais_decoder = HeterogeneousAttributeDecoder(self.ais_encoding_dim, dataset_stats).to(cfg.device)

        self.loss_calculator = loss_calculator

    def __str__(self):
        return "simple"

    def forward(self, ais_batch: AISBatch, curric_prob: float = 0) -> tuple[LossTypes, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        true_lats = ais_batch.lats.to(self._cfg.device)
        true_lons = ais_batch.lons.to(self._cfg.device)
        lats = true_lats.clone().contiguous().to(self._cfg.device)
        lons = true_lons.clone().contiguous().to(self._cfg.device)
        timestamps = ais_batch.observed_timestamps.contiguous().to(self._cfg.device)
        observed = ais_batch.observed_data.contiguous().to(self._cfg.device)
        masks = ais_batch.masks.to(self._cfg.device)
        fine_masks = torch.repeat_interleave(masks, self._cfg.dim_ais_attr_encoding, dim=2).detach()

        all_deltas_pred = []
        all_deltas_true = []
        all_pos_pred = []
        all_pos_true = []

        encoded = self.ais_encoder(observed)
        b, s, f = encoded.size()

        total_consistency_loss = torch.tensor(0.0, device=self._cfg.device)

        assert ais_batch.num_missing_values % 2 == 0, "Number of missing values must be even."

        for i in range(ais_batch.num_missing_values // 2):

            brits_data = _prepare_brits_data(timestamps, encoded, fine_masks)

            brits_ret = self.impu_module(brits_data, stage="test")
            imputed = brits_ret["imputed_data"][:, :, :self.ais_encoding_dim]
            total_consistency_loss += brits_ret["consistency_loss"]

            # find first missing timestep per batch
            first_mask_idx = (fine_masks == 0).any(dim=2).float().argmax(dim=1)
            last_mask_idx = s - 1 - (fine_masks == 0).flip(dims=[1]).any(dim=2).float().argmax(dim=1)

            batch_idx = torch.arange(b, device=imputed.device)

            first_imputed = imputed[batch_idx, first_mask_idx, :]
            first_encoded = encoded[batch_idx, first_mask_idx, :]
            first_truth = observed[batch_idx, first_mask_idx, :]

            last_imputed = imputed[batch_idx, last_mask_idx, :]
            last_encoded = encoded[batch_idx, last_mask_idx, :]
            last_truth = observed[batch_idx, last_mask_idx, :]

            # teacher forcing
            if self.training:
                first_tf_mask = (torch.rand(b, device=first_imputed.device) < curric_prob).float().unsqueeze(-1)
                last_tf_mask = (torch.rand(b, device=last_imputed.device) < curric_prob).float().unsqueeze(-1)

                first_input = first_tf_mask * first_encoded + (1 - first_tf_mask) * first_imputed
                last_input = last_tf_mask * last_encoded + (1 - last_tf_mask) * last_imputed
            else:
                first_input = first_imputed
                last_input = last_imputed

            first_input = first_input.unsqueeze(1)
            first_truth = first_truth.unsqueeze(1)
            first_encoded = first_encoded.unsqueeze(1)

            last_input = last_input.unsqueeze(1)
            last_truth = last_truth.unsqueeze(1)
            last_encoded = last_encoded.unsqueeze(1)

            # decode
            first_decoded = self.ais_decoder(first_input)
            last_decoded = self.ais_decoder(last_input)

            eastern_deltas_first = first_decoded[batch_idx, :, AISColDict.EASTERN_DELTA.value].detach()
            northern_deltas_first = first_decoded[batch_idx, :, AISColDict.NORTHERN_DELTA.value].detach()
            eastern_deltas_last = last_decoded[batch_idx, :, AISColDict.EASTERN_DELTA.value].detach()
            northern_deltas_last = last_decoded[batch_idx, :, AISColDict.NORTHERN_DELTA.value].detach()

            # udpate lat lons based on deltas
            self.update_lat_lon(lats, lons, eastern_deltas_first, northern_deltas_first,
                                first_mask_idx, direction="forward")
            self.update_lat_lon(lats, lons, eastern_deltas_last, northern_deltas_last,
                                last_mask_idx, direction="backward")

            first_lat_pred = lats[batch_idx, first_mask_idx]
            first_lon_pred = lons[batch_idx, first_mask_idx]
            last_lat_pred = lats[batch_idx, last_mask_idx]
            last_lon_pred = lons[batch_idx, last_mask_idx]

            first_lat_truth = true_lats[batch_idx, first_mask_idx]
            first_lon_truth = true_lons[batch_idx, first_mask_idx]
            last_lat_truth = true_lats[batch_idx, last_mask_idx]
            last_lon_truth = true_lons[batch_idx, last_mask_idx]

            first_pos_pred = torch.stack([first_lat_pred, first_lon_pred], dim=-1).unsqueeze(1)
            last_pos_pred = torch.stack([last_lat_pred, last_lon_pred], dim=-1).unsqueeze(1)
            first_pos_true = torch.stack([first_lat_truth, first_lon_truth], dim=-1).unsqueeze(1)
            last_pos_true = torch.stack([last_lat_truth, last_lon_truth], dim=-1).unsqueeze(1)

            # update encoded sequence with NEW ground truth / imputed value
            first_scatter_index = first_mask_idx.view(-1, 1, 1).expand(-1, 1, f)
            last_scatter_index = last_mask_idx.view(-1, 1, 1).expand(-1, 1, f)

            encoded = encoded.scatter(1, first_scatter_index, first_encoded.detach())
            encoded = encoded.scatter(1, last_scatter_index, last_encoded.detach())

            fine_masks = fine_masks.scatter(1, first_scatter_index, 1)
            fine_masks = fine_masks.scatter(1, last_scatter_index, 1)

            all_pos_pred.insert(i, first_pos_pred)
            all_pos_true.insert(i, first_pos_true)
            all_deltas_pred.insert(i, first_decoded)
            all_deltas_true.insert(i, first_truth)

            all_pos_pred.insert(i + 1, last_pos_pred)
            all_pos_true.insert(i + 1, last_pos_true)
            all_deltas_pred.insert(i + 1, last_decoded)
            all_deltas_true.insert(i + 1, last_truth)

        # concatenate all decoded steps
        all_deltas_pred = torch.cat(all_deltas_pred, dim=1)
        all_deltas_true = torch.cat(all_deltas_true, dim=1)
        all_pos_pred = torch.cat(all_pos_pred, dim=1)
        all_pos_true = torch.cat(all_pos_true, dim=1)

        forces_pred = torch.zeros(1, device=all_pos_pred.device)
        forces_true = torch.zeros(1, device=all_pos_true.device)

        full_traj_pred = torch.stack([lats, lons], dim=-1)
        full_traj_true = torch.stack([true_lats, true_lons], dim=-1)

        loss = self.loss_calculator.calculate_loss(
            full_traj_pred, full_traj_true,
            all_pos_pred, all_pos_true,
            all_deltas_pred, all_deltas_true,
            total_consistency_loss,
            forces_pred, forces_true)

        return loss, (lats, lons, true_lats, true_lons)

    def update_lat_lon(self, lats: torch.Tensor, lons: torch.Tensor, eastern_deltas: torch.Tensor,
                       northern_deltas: torch.Tensor, mask_indices: torch.Tensor, direction: str) -> None:
        b, s = lats.size()
        batch_idx = torch.arange(b, device=lats.device)
        if direction == "forward":
            prev_mask_indices = mask_indices - 1
            lats_to_update = lats[batch_idx, prev_mask_indices]
            lons_to_update = lons[batch_idx, prev_mask_indices]
            E, N = GC.espg4326_to_epsg3034_batch_tensor(lons_to_update, lats_to_update)
            E += eastern_deltas.squeeze(-1)
            N += northern_deltas.squeeze(-1)
            lons_updated, lats_updated = GC.epsg3034_to_espg4326_batch_tensor(E, N)
            lats[batch_idx, mask_indices] = lats_updated
            lons[batch_idx, mask_indices] = lons_updated

        elif direction == "backward":
            prev_mask_indices = mask_indices + 1
            lats_to_update = lats[batch_idx, prev_mask_indices]
            lons_to_update = lons[batch_idx, prev_mask_indices]
            E, N = GC.espg4326_to_epsg3034_batch_tensor(lons_to_update, lats_to_update)
            E -= eastern_deltas.squeeze(-1)
            N -= northern_deltas.squeeze(-1)
            lons_updated, lats_updated = GC.epsg3034_to_espg4326_batch_tensor(E, N)
            lats[batch_idx, mask_indices] = lats_updated
            lons[batch_idx, mask_indices] = lons_updated


def _prepare_brits_data(timestamps, encoded_data, masks):
    b, s, f = encoded_data.size()

    def compute_deltas(ts, ms):
        delta_data = torch.zeros((b, s, f), device=encoded_data.device).detach()

        for t in range(1, s):
            time_gap = (ts[:, t] - ts[:, t - 1]).unsqueeze(-1).repeat_interleave(f, dim=1)
            previous_mask = ms[:, t - 1, :]
            reset_deltas = time_gap * previous_mask
            accumulated_deltas = (time_gap + delta_data[:, t - 1, :]) * (1 - previous_mask)
            delta_data[:, t, :] = reset_deltas + accumulated_deltas
        return delta_data

    forward_deltas = compute_deltas(timestamps, masks)
    flipped_masks = torch.flip(masks, [1])
    backward_deltas = torch.abs(compute_deltas(torch.flip(timestamps, [1]), flipped_masks))

    return {
        "forward": {
            "X": encoded_data,
            "missing_mask": masks,
            "deltas": forward_deltas,
        },
        "backward": {
            "X": torch.flip(encoded_data, [1]),
            "missing_mask": flipped_masks,
            "deltas": backward_deltas,
        },
    }
