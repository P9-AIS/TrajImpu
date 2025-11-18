import torch
import torch.nn as nn
from dataclasses import dataclass
from ForceProviders.i_force_provider import IForceProvider
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.afa_module import AFAModule
from Model.brits import BRITS
from Model.ais_decoder import ExtraDecodeOutput, HeterogeneousAttributeDecoder
from ModelTypes.ais_dataset_masked import AISBatch
from ModelTypes.ais_stats import AISStats
from ModelUtils.loss_calculator import LossCalculator, LossOutput


@dataclass
class Config:
    device: str
    teacher_forcing_ratio: float = 0.5

    # encoder
    dim_ais_attr_encoding: int = 16

    # afa module
    num_layers: int = 2
    num_head: int = 4

    # brits
    dim_rnn_hidden: int = 10
    MIT: bool = True

    # decoder
    dim_hidden: int = 64


class Model(nn.Module):
    def __init__(self, dataset_stats: AISStats, force_providers: list[IForceProvider],
                 loss_calculator: LossCalculator, cfg: Config):

        super().__init__()
        self._cfg = cfg

        self.ais_encoder = HeterogeneousAttributeEncoder(
            cfg.dim_ais_attr_encoding, dataset_stats).to(cfg.device)

        self.ais_encoding_dim = self.ais_encoder.output_dim

        self.afa_module = AFAModule(self.ais_encoding_dim, cfg.num_head, *force_providers).to(cfg.device)

        self.impu_module = BRITS(
            dataset_stats.seq_len, self.ais_encoding_dim,
            cfg.dim_rnn_hidden, MIT=cfg.MIT, device=cfg.device
        ).to(cfg.device)

        self.ais_decoder = HeterogeneousAttributeDecoder(
            self.ais_encoding_dim, dataset_stats, self.ais_encoding_dim
        ).to(cfg.device)

        self.loss_calculator = loss_calculator

    def forward(self, ais_batch: AISBatch) -> LossOutput:
        observed = ais_batch.observed_data.contiguous().to(self._cfg.device)
        masks = ais_batch.masks.to(self._cfg.device)

        # Encode sequence (full gradient)
        encoded = self.ais_encoder(observed)

        # Expand mask across features — masks don’t have gradients anyway
        fine_masks = torch.repeat_interleave(masks, self._cfg.dim_ais_attr_encoding, dim=2).detach()

        all_decoded = []
        all_decoded_extra = []
        all_truth = []

        for _ in range(ais_batch.num_missing_values):

            # ---- BRITS INPUT (IMPORTANT: do NOT detach encoded data!) ----
            brits_data = _prepare_brits_data(observed, encoded, fine_masks)

            imputed = self.impu_module(brits_data, stage="test")["imputed_data"]

            # find first missing timestep per batch
            first_mask_idx = (fine_masks == 0).any(dim=2).float().argmax(dim=1)

            batch_idx = torch.arange(imputed.size(0), device=imputed.device)

            first_imputed = imputed[batch_idx, first_mask_idx, :]
            first_encoded = encoded[batch_idx, first_mask_idx, :]
            first_truth = observed[batch_idx, first_mask_idx, :]

            # teacher forcing
            if self.training:
                tf_mask = (torch.rand(batch_idx.size(0), device=first_imputed.device)
                           < self._cfg.teacher_forcing_ratio).float().unsqueeze(-1)

                # do NOT detach here
                first_input = tf_mask * first_encoded + (1 - tf_mask) * first_imputed
            else:
                first_input = first_imputed

            # make sequence dimension 1
            first_input = first_input.unsqueeze(1)
            first_truth = first_truth.unsqueeze(1)
            first_encoded = first_encoded.unsqueeze(1)

            # decode
            first_decoded, extra = self.ais_decoder(first_input)

            # update encoded sequence with NEW ground truth / imputed value
            scatter_index = first_mask_idx.view(-1, 1, 1).expand(-1, 1, encoded.size(2))

            # This detach breaks the autoregressive graph – CORRECT
            encoded = encoded.scatter(1, scatter_index, first_encoded.detach())
            fine_masks = fine_masks.scatter(1, scatter_index, 1)

            all_decoded.append(first_decoded)
            all_decoded_extra.append(extra)
            all_truth.append(first_truth)

        # concatenate all decoded steps
        all_decoded_tensor = torch.cat(all_decoded, dim=1)
        all_truth_tensor = torch.cat(all_truth, dim=1)

        loss = self.loss_calculator.calculate_loss(
            all_decoded_tensor, all_decoded_extra, all_truth_tensor
        )

        return loss


# --------------------------------------------------------------------
# BRITS data preparation — FIXED
# --------------------------------------------------------------------
def _prepare_brits_data(ais_data, encoded_data, masks):
    """
    IMPORTANT:
    - encoded_data must NOT be detached (BRITS must learn from it)
    - masks/deltas can be detached or not; they don't need gradients
    """

    b, s, f = encoded_data.size()

    delta_data = torch.zeros((b, s, f), device=encoded_data.device).detach()

    for t in range(1, s):
        time_gap = (ais_data[:, t, 0] - ais_data[:, t - 1, 0]).unsqueeze(-1)
        previous_mask = masks[:, t - 1, :]
        reset_deltas = time_gap * previous_mask
        accumulated_deltas = (time_gap + delta_data[:, t - 1, :]) * (1 - previous_mask)
        delta_data[:, t, :] = reset_deltas + accumulated_deltas

    return {
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
