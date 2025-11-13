import torch
import torch.nn as nn
from dataclasses import field, dataclass
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ForceProviders.i_force_provider import IForceProvider
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.afa_module import AFAModule
from Model.brits import BRITS
from Model.ais_decoder import HeterogeneousAttributeDecoder
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
            cfg.dim_ais_attr_encoding, dataset_stats, cfg.max_delta).to(cfg.device)

        self._cfg = cfg
        self.ais_encoding_dim = self.ais_encoder.output_dim

        self.afa_module = AFAModule(self.ais_encoding_dim, cfg.num_head, *force_providers).to(cfg.device)
        self.impu_module = BRITS(dataset_stats.seq_len, self.ais_encoding_dim,
                                 cfg.dim_rnn_hidden, MIT=cfg.MIT, device=cfg.device).to(cfg.device)
        self.ais_decoder = HeterogeneousAttributeDecoder(
            self.ais_encoding_dim, dataset_stats, self.ais_encoding_dim).to(cfg.device)

    def forward(self, ais_batch: AISBatch):
        ais_batch.observed_data = ais_batch.observed_data.contiguous().to(self._cfg.device)
        ais_batch.masks = ais_batch.masks.to(self._cfg.device)

        encoded = self.ais_encoder(ais_batch.observed_data)
        features, _ = self.afa_module(ais_batch.observed_data, encoded)

        fine_masks = torch.repeat_interleave(ais_batch.masks, self._cfg.dim_ais_attr_encoding, dim=2)
        current_masks = fine_masks.clone()

        all_imputed = []
        all_imputed_truth = []

        for _ in range(ais_batch.num_missing_values):
            brits_data = _prepare_brits_data(ais_batch.observed_data, features, current_masks)

            imputed = self.impu_module(brits_data, stage="test")["imputed_data"]  # [b, s, feat_dim]

            # Select index of first missing value per batch
            first_mask_idx = (current_masks == 0).any(dim=2).float().argmax(dim=1)  # [b]

            # Gather imputed feature for that position
            batch_idx = torch.arange(imputed.size(0), device=imputed.device)
            first_imputed = imputed[batch_idx, first_mask_idx, :]                  # [b, feat_dim]
            first_imputed_truth = ais_batch.observed_data[batch_idx, first_mask_idx, :]

            # Decode -> enrich -> re-encode
            first_decoded = self.ais_decoder(first_imputed.unsqueeze(1))           # [b, 1, num_ais_attr]
            first_encoded = self.ais_encoder(first_decoded)                        # [b, 1, feat_dim]
            first_features, _ = self.afa_module(first_decoded, first_encoded)      # [b, 1, feat_dim]

            # Replace feature at that timestep in a differentiable way
            scatter_index = first_mask_idx.view(-1, 1, 1).expand(-1, 1, features.size(2))
            features = features.clone()
            features.scatter_(1, scatter_index, first_features)

            # Mark mask as filled
            current_masks = current_masks.clone()
            current_masks[batch_idx, first_mask_idx, :] = 1

            all_imputed.append(first_features)
            all_imputed_truth.append(first_imputed_truth.unsqueeze(1))

        all_imputed = torch.cat(all_imputed, dim=1)
        all_imputed_truth = torch.cat(all_imputed_truth, dim=1)

        decoded_imputed = self.ais_decoder(all_imputed)
        loss_total, loss_tuple = compute_loss(decoded_imputed, all_imputed_truth)

        return loss_total, loss_tuple


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


def compute_loss(predictions: torch.Tensor, truths: torch.Tensor):
    loss_mae = nn.L1Loss()(predictions, truths)
    epsilon = 1e-6
    smape_numerator = torch.abs(predictions - truths)
    smape_denominator = (torch.abs(predictions) + torch.abs(truths)) + epsilon
    loss_smape = torch.mean(smape_numerator / smape_denominator)

    alpha, beta = 0.5, 0.5
    loss_total = alpha * loss_mae + beta * loss_smape

    return loss_total, (loss_mae, loss_smape)
