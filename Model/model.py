import torch
import torch.nn as nn
from dataclasses import field, dataclass
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ForceProviders.i_force_provider import IForceProvider
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.afa_module import AFAModule
from Model.brits import BRITS
from Model.ais_decoder import AISDecoder
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_stats import AISStats


@dataclass
class Config:
    device: str

    # encoder
    dim_ais_attr_encoding: int = 16

    max_delta: float = 0.0

    # afa module
    num_layers: int = 2
    num_head: int = 4

    # brits
    dim_rnn_hidden: int = 10
    MIT: bool = True

    # decoder
    dim_hidden: int = 64


class Model(nn.Module):
    def __init__(self, dataset_stats: AISStats, force_providers: list[IForceProvider], cfg: Config):
        super().__init__()
        self.ais_encoder = HeterogeneousAttributeEncoder(
            cfg.dim_ais_attr_encoding,
            dataset_stats,
            cfg.max_delta,
        )
        self._cfg = cfg
        self.ais_encoding_dim = self.ais_encoder.output_dim

        self.afa_module = AFAModule(self.ais_encoding_dim, cfg.num_head, *force_providers)
        self.impu_module = BRITS(200, self.ais_encoding_dim, 13, MIT=cfg.MIT, device=cfg.device)
        self.ais_decoder = AISDecoder(self.ais_encoding_dim, dataset_stats.num_attributes)

    def forward(self, ais_batch: AISBatch):
        encoded = self.ais_encoder(ais_batch.observed_data)
        features, _ = self.afa_module(ais_batch.observed_data, encoded)

        current_ais_data = ais_batch.observed_data.clone()
        current_masks = torch.repeat_interleave(ais_batch.masks, self._cfg.dim_ais_attr_encoding, dim=2)

        for _ in range(ais_batch.num_missing_values):
            brits_data = _prepare_brits_data(current_ais_data, features, current_masks)

            imputed = self.impu_module(brits_data, stage="test")["imputed_data"]  # shape [b, s, feature_dim]

            first_mask_idx = (current_masks == 0).any(dim=2).float().argmax(dim=1)  # shape [b]
            first_imputed = imputed[torch.arange(imputed.size(0)), first_mask_idx, :]  # shape [b, feature_dim]
            first_imputed = first_imputed.unsqueeze(1)  # [b, 1, feature_dim]
            first_decoded = self.ais_decoder(first_imputed)  # shape [b, 1, num_ais_attributes]
            first_encoded = self.ais_encoder(first_decoded)  # shape [b, 1, feature_dim]
            first_features, _ = self.afa_module(first_decoded, first_encoded)  # shape [b, 1, feature_dim]

            features[torch.arange(features.size(0)), first_mask_idx, :] = first_features.squeeze(1)
            current_masks[torch.arange(current_masks.size(0)), first_mask_idx, :] = 1

        final = self.ais_decoder(features)
        loss_total, loss_list = compute_loss(final, ais_batch, ais_batch.masks)

        return loss_total, loss_list


def _prepare_brits_data(ais_data, encoded_data, masks):
    b, s, f = encoded_data.size()
    delta_data = torch.zeros((b, s, f), device=encoded_data.device)

    for t in range(1, s):
        time_gap = (ais_data[:, t, 0] - ais_data[:, t - 1, 0]).unsqueeze(-1)  # [b, 1]
        previous_mask = masks[:, t - 1, :]  # [b, f]
        reset_deltas = time_gap * previous_mask  # [b, f]
        accumulated_deltas = (time_gap + delta_data[:, t - 1, :]) * (1 - previous_mask)  # [b, f]
        delta_data[:, t, :] = reset_deltas + accumulated_deltas

    data = {
        "forward": {
            "X": encoded_data,
            "missing_mask": masks,
            "deltas": delta_data,
        },
        "backward": {
            "X": torch.flip(encoded_data, [1]),
            "missing_mask": torch.flip(masks, [1]),
            "deltas": torch.flip(delta_data, [1]),
        },
    }
    return data


def compute_loss(predictions, targets, masks):
    negated_masks = 1 - masks
    loss_mae = torch.sum(torch.abs(predictions - targets) * negated_masks) / (torch.sum(negated_masks) + 1e-9)

    loss_smape = torch.sum(torch.abs(predictions - targets) * negated_masks) / \
        (torch.sum(torch.abs(targets) + torch.abs(predictions)) / 2 + 1e-9)

    return loss_mae, loss_smape
