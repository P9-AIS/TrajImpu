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
from ModelUtils.loss_calculator import LossCalculator, LossOutput, LossTypes
from ForceUtils.geo_converter import GeoConverter as GC


@dataclass
class Config:
    device: str
    teacher_forcing_ratio: float

    # encoder
    dim_ais_attr_encoding: int

    # afa module
    num_layers: int
    num_head: int

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
        return "force_model"

    def forward(self, ais_batch: AISBatch) -> tuple[LossTypes, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        true_lats = ais_batch.lats.to(self._cfg.device)
        true_lons = ais_batch.lons.to(self._cfg.device)
        lats = true_lats.clone().contiguous().to(self._cfg.device)
        lons = true_lons.clone().contiguous().to(self._cfg.device)
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

        all_decoded = []
        all_truth = []

        encoded = self.ais_encoder(observed)
        forces, true_forces = self.force_encoder(lats, lons)
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

            # teacher forcing
            if self.training:
                tf_mask = (torch.rand(b, device=first_imputed.device) <
                           self._cfg.teacher_forcing_ratio).float().unsqueeze(-1)
                first_input = tf_mask * first_encoded + (1 - tf_mask) * first_imputed
            else:
                first_input = first_imputed

            if self.training:
                tf_mask = (torch.rand(b, device=first_imputed.device) <
                           self._cfg.teacher_forcing_ratio).float().unsqueeze(-1)
                last_input = tf_mask * last_encoded + (1 - tf_mask) * last_imputed
            else:
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

            # get updated lat lons
            first_lat = lats[batch_idx, first_mask_idx]
            first_lon = lons[batch_idx, first_mask_idx]
            last_lat = lats[batch_idx, last_mask_idx]
            last_lon = lons[batch_idx, last_mask_idx]

            first_forces, _ = self.force_encoder(first_lat.unsqueeze(1), first_lon.unsqueeze(1))
            last_forces, _ = self.force_encoder(last_lat.unsqueeze(1), last_lon.unsqueeze(1))

            first_features = torch.cat((first_encoded, first_forces), dim=-1)
            last_features = torch.cat((last_encoded, last_forces), dim=-1)

            # update encoded sequence with NEW ground truth / imputed value
            first_scatter_index = first_mask_idx.view(-1, 1, 1).expand(-1, 1, f)
            last_scatter_index = last_mask_idx.view(-1, 1, 1).expand(-1, 1, f)

            features = features.scatter(1, first_scatter_index, first_features.detach())
            features = features.scatter(1, last_scatter_index, last_features.detach())

            fine_masks = fine_masks.scatter(1, first_scatter_index, 1)
            fine_masks = fine_masks.scatter(1, last_scatter_index, 1)

            all_decoded.insert(i, first_decoded)
            all_truth.insert(i, first_truth)

            all_decoded.insert(i + 1, last_decoded)
            all_truth.insert(i + 1, last_truth)

        # concatenate all decoded steps
        all_decoded_tensor = torch.cat(all_decoded, dim=1)
        all_truth_tensor = torch.cat(all_truth, dim=1)

        decoded_forces = self.force_decoder(forces)

        loss = self.loss_calculator.calculate_loss(
            all_decoded_tensor, all_truth_tensor, total_consistency_loss, decoded_forces, true_forces)

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
