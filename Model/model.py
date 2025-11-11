import torch
import torch.nn as nn
from dataclasses import field, dataclass
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.afa_module import AFAModule
from Model.brits import BRITS
from Model.ais_decoder import AISDecoder
from ModelTypes.ais_dataset_masked import AISBatch


@dataclass
class Config:
    device: str

    # encoder
    dim_ais_attr_encoding: int = 16
    status: str = "test"
    num_ais_attributes: int = 14
    num_navi_status_class: int = 0
    num_destination_class: int = 0
    num_cargo_type_class: int = 0
    num_vessel_type_class: int = 0
    max_delta: float = 0.0

    # afa module
    num_layers: int = 2
    num_head: int = 4

    # brits
    seq_len: int = 64
    dim_rnn_hidden: int = 10
    MIT: bool = True

    # decoder
    dim_hidden: int = 64


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ais_encoder = HeterogeneousAttributeEncoder(
            cfg.dim_ais_attr_encoding,
            cfg.status,
            cfg.num_navi_status_class,
            cfg.num_destination_class,
            cfg.num_cargo_type_class,
            cfg.num_vessel_type_class,
            cfg.max_delta,
        )

        self.ais_encoding_dim = cfg.dim_ais_attr_encoding * cfg.num_ais_attributes

        self.afa_module = AFAModule(self.ais_encoding_dim, cfg.num_head, cfg.num_layers)
        self.impu_module = BRITS(cfg.seq_len, self.ais_encoding_dim + 3,
                                 cfg.dim_rnn_hidden, MIT=cfg.MIT, device=cfg.device)
        self.ais_decoder = AISDecoder(self.ais_encoding_dim + 3, cfg.num_ais_attributes)

    def forward(self, ais_batch: AISBatch):
        encoded = self.ais_encoder(ais_batch)
        forces, _ = self.afa_module(encoded)
        features = torch.cat((encoded, forces), dim=2)

        current_ais_data = ais_batch.observed_data.clone()
        current_masks = ais_batch.masks.clone()

        for _ in range(ais_batch.num_missing_values):
            brits_data = prepare_brits_data(current_ais_data, features, current_masks)

            imputed = self.impu_module(brits_data, stage="test")["imputed_data"]

            first_mask_idx = (current_masks == 0).any(dim=2).float().argmax(dim=1)
            first_imputed = imputed[torch.arange(imputed.size(0)), first_mask_idx, :]
            first_decoded = self.ais_decoder(first_imputed)
            first_encoded = self.ais_encoder(first_decoded)
            first_forces, _ = self.afa_module(first_encoded)
            first_features = torch.cat((first_encoded, first_forces), dim=2)

            features[torch.arange(ais_batch.num_values_in_sequence), first_mask_idx, :] = first_features
            current_masks[torch.arange(ais_batch.num_values_in_sequence), first_mask_idx, :] = 1

        final = self.ais_decoder(features)
        loss_total, loss_list = compute_loss(final, ais_batch, ais_batch.masks)

        return loss_total, loss_list


def prepare_brits_data(ais_data, encoded_data, masks):
    b, s, f = encoded_data.size()
    delta_data = torch.zeros((b, s, f), device=encoded_data.device)

    for i in range(1, s):
        time_gap = ais_data[:, i, 0] - ais_data[:, i - 1, 0]
        time_gap = time_gap.unsqueeze(-1)
        delta_data[:, i, :] = (delta_data[:, i - 1, :] + time_gap) * (1 - masks[:, i - 1, :])

    data = {
        "X": encoded_data,
        "missing_mask": masks,
        "deltas": delta_data,
    }
    return data


def compute_loss(predictions, targets, masks):
    negated_masks = 1 - masks
    loss_mae = torch.sum(torch.abs(predictions - targets) * negated_masks) / (torch.sum(negated_masks) + 1e-9)

    loss_smape = torch.sum(torch.abs(predictions - targets) * negated_masks) / \
        (torch.sum(torch.abs(targets) + torch.abs(predictions)) / 2 + 1e-9)

    return loss_mae, loss_smape
