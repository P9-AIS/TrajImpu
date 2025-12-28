import torch
import torch.nn as nn
from dataclasses import dataclass
from ForceProviders.i_force_provider import IForceProvider
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.force_decoder import ForceDecoder
from Model.force_encoder import ForceEncoder
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
    def __init__(self, dataset_stats: AISStats, force_provider: IForceProvider,
                 loss_calculator: LossCalculator, cfg: Config):

        super().__init__()
        self._cfg = cfg

        self.ais_encoder = HeterogeneousAttributeEncoder(cfg.dim_ais_attr_encoding, dataset_stats).to(cfg.device)
        self.force_encoder = ForceEncoder(cfg.dim_ais_attr_encoding, force_provider).to(cfg.device)

        self.ais_encoding_dim = self.ais_encoder.output_dim
        self.force_encoding_dim = cfg.dim_ais_attr_encoding
        self.feature_encoding_dim = self.ais_encoding_dim + self.force_encoding_dim

        self.impu_module = BRITS(
            dataset_stats.seq_len, self.feature_encoding_dim,
            cfg.dim_rnn_hidden, MIT=cfg.MIT, device=cfg.device
        ).to(cfg.device)

        self.ais_decoder = HeterogeneousAttributeDecoder(self.ais_encoding_dim, dataset_stats).to(cfg.device)
        self.force_decoder = ForceDecoder(self.force_encoding_dim).to(cfg.device)

        self.loss_calculator = loss_calculator

    def __str__(self):
        return "force"

    def forward(self, ais_batch: AISBatch, curric_prob: float = 0) -> tuple[LossTypes, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        northerns = ais_batch.northerns.contiguous().to(self._cfg.device)
        easterns = ais_batch.easterns.contiguous().to(self._cfg.device)
        true_northerns = northerns.contiguous().clone().to(self._cfg.device).detach()
        true_easterns = easterns.contiguous().clone().to(self._cfg.device).detach()
        timestamps = ais_batch.observed_timestamps.contiguous().to(self._cfg.device)
        observed = ais_batch.observed_data.contiguous().to(self._cfg.device)

        masks = ais_batch.masks.to(self._cfg.device)
        mask_north = masks[:, :, AISColDict.NORTHERN_DELTA.value].unsqueeze(-1)
        mask_east = masks[:, :, AISColDict.EASTERN_DELTA.value].unsqueeze(-1)
        mask_force_base = mask_north * mask_east
        fine_mask_force = mask_force_base.repeat(1, 1, self._cfg.dim_ais_attr_encoding)
        fine_mask_north = mask_north.repeat(1, 1, self._cfg.dim_ais_attr_encoding)
        fine_mask_east = mask_east.repeat(1, 1, self._cfg.dim_ais_attr_encoding)
        fine_masks = torch.cat([
            fine_mask_north,
            fine_mask_east,
            fine_mask_force
        ], dim=-1).detach()

        all_deltas_pred = []
        all_deltas_true = []
        all_pos_pred = []
        all_pos_true = []

        encoded = self.ais_encoder(observed)
        forces, forces_true = self.force_encoder(northerns, easterns)
        features = torch.cat((encoded, forces), dim=-1)  # shape [b, s, e * 3]

        b, s, f = features.size()
        total_consistency_loss = torch.tensor(0.0, device=self._cfg.device)

        assert ais_batch.num_missing_values % 2 == 0, "Number of missing values must be even."

        for i in range(ais_batch.num_missing_values // 2):

            brits_data = _prepare_brits_data(timestamps, features, fine_masks)

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

            # curriculum learning
            if self.training:
                first_tf_mask = (torch.rand(b, device=first_imputed.device) < curric_prob).float().unsqueeze(-1)
                last_tf_mask = (torch.rand(b, device=last_imputed.device) < curric_prob).float().unsqueeze(-1)

                last_input = last_tf_mask * last_encoded + (1 - last_tf_mask) * last_imputed
                first_input = first_tf_mask * first_encoded + (1 - first_tf_mask) * first_imputed
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

            eastern_deltas_first = first_decoded[batch_idx, :, AISColDict.EASTERN_DELTA.value]
            northern_deltas_first = first_decoded[batch_idx, :, AISColDict.NORTHERN_DELTA.value]
            eastern_deltas_last = last_decoded[batch_idx, :, AISColDict.EASTERN_DELTA.value]
            northern_deltas_last = last_decoded[batch_idx, :, AISColDict.NORTHERN_DELTA.value]

            # udpate lat lons based on deltas
            self.update_lat_lon(easterns, northerns, eastern_deltas_first, northern_deltas_first,
                                first_mask_idx, direction="forward")
            self.update_lat_lon(easterns, northerns, eastern_deltas_last, northern_deltas_last,
                                last_mask_idx, direction="backward")

            # get updated lat lons
            first_northern_pred = northerns[batch_idx, first_mask_idx]
            first_eastern_pred = easterns[batch_idx, first_mask_idx]
            last_northern_pred = northerns[batch_idx, last_mask_idx]
            last_eastern_pred = easterns[batch_idx, last_mask_idx]

            first_northern_truth = true_northerns[batch_idx, first_mask_idx]
            first_eastern_truth = true_easterns[batch_idx, first_mask_idx]
            last_northern_truth = true_northerns[batch_idx, last_mask_idx]
            last_eastern_truth = true_easterns[batch_idx, last_mask_idx]

            first_pos_pred = torch.stack([first_northern_pred, first_eastern_pred], dim=-1).unsqueeze(1)
            last_pos_pred = torch.stack([last_northern_pred, last_eastern_pred], dim=-1).unsqueeze(1)
            first_pos_true = torch.stack([first_northern_truth, first_eastern_truth], dim=-1).unsqueeze(1)
            last_pos_true = torch.stack([last_northern_truth, last_eastern_truth], dim=-1).unsqueeze(1)

            first_forces, _ = self.force_encoder(first_northern_pred.clone().unsqueeze(
                1).detach(), first_eastern_pred.clone().unsqueeze(1).detach())
            last_forces, _ = self.force_encoder(last_northern_pred.clone().unsqueeze(
                1).detach(), last_eastern_pred.clone().unsqueeze(1).detach())

            first_features = torch.cat((first_encoded, first_forces), dim=-1)
            last_features = torch.cat((last_encoded, last_forces), dim=-1)

            # update encoded sequence with NEW ground truth / imputed value
            first_scatter_index = first_mask_idx.view(-1, 1, 1).expand(-1, 1, f)
            last_scatter_index = last_mask_idx.view(-1, 1, 1).expand(-1, 1, f)

            features = features.scatter(1, first_scatter_index, first_features.detach())
            features = features.scatter(1, last_scatter_index, last_features.detach())

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

        forces_pred = self.force_decoder(forces)

        full_traj_pred = torch.stack([northerns, easterns], dim=-1)
        full_traj_true = torch.stack([true_northerns, true_easterns], dim=-1)

        loss = self.loss_calculator.calculate_loss(
            self.training,
            full_traj_pred, full_traj_true,
            all_pos_pred, all_pos_true,
            all_deltas_pred, all_deltas_true,
            total_consistency_loss,
            forces_pred, forces_true)

        return loss, (northerns, easterns, true_northerns, true_easterns)

    def update_lat_lon(self, easterns: torch.Tensor, northerns: torch.Tensor, eastern_deltas: torch.Tensor,
                       northern_deltas: torch.Tensor, mask_indices: torch.Tensor, direction: str) -> None:
        b, s = northerns.size()
        batch_idx = torch.arange(b, device=northerns.device)
        if direction == "forward":
            prev_mask_indices = mask_indices - 1
            northerns_to_update = northerns[batch_idx, prev_mask_indices]
            easterns_to_update = easterns[batch_idx, prev_mask_indices]

            easterns_to_update += eastern_deltas.squeeze(-1)
            northerns_to_update += northern_deltas.squeeze(-1)

            northerns[batch_idx, mask_indices] = northerns_to_update
            easterns[batch_idx, mask_indices] = easterns_to_update

        elif direction == "backward":
            prev_mask_indices = mask_indices + 1
            northerns_to_update = northerns[batch_idx, prev_mask_indices]
            easterns_to_update = easterns[batch_idx, prev_mask_indices]

            easterns_to_update -= eastern_deltas.squeeze(-1)
            northerns_to_update -= northern_deltas.squeeze(-1)

            northerns[batch_idx, mask_indices] = northerns_to_update
            easterns[batch_idx, mask_indices] = easterns_to_update


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
