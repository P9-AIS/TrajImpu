import torch
import torch.nn as nn
from dataclasses import dataclass
from Model.ais_encoder import HeterogeneousAttributeEncoder
from Model.afa_module import AFAModule
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
    dim_ais_attr_encoding: int


class Model(nn.Module):
    def __init__(self, dataset_stats: AISStats, loss_calculator: LossCalculator, cfg: Config):

        super().__init__()
        self._cfg = cfg

        self.ais_encoder = HeterogeneousAttributeEncoder(cfg.dim_ais_attr_encoding, dataset_stats).to(cfg.device)

        self.ais_encoding_dim = self.ais_encoder.output_dim

        self.ais_decoder = HeterogeneousAttributeDecoder(self.ais_encoding_dim, dataset_stats).to(cfg.device)

        self.loss_calculator = loss_calculator

    def __str__(self):
        return "enc_dec_model"

    def forward(self, ais_batch: AISBatch) -> tuple[LossTypes, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        true_lats = ais_batch.lats.to(self._cfg.device)
        true_lons = ais_batch.lons.to(self._cfg.device)

        observed = ais_batch.observed_data.contiguous().to(self._cfg.device)

        encoded = self.ais_encoder(observed)
        decoded = self.ais_decoder(encoded)

        lats = decoded[:, :, AISColDict.NORTHERN_DELTA.value:AISColDict.NORTHERN_DELTA.value+1]
        lons = decoded[:, :, AISColDict.EASTERN_DELTA.value:AISColDict.EASTERN_DELTA.value+1]

        loss = self.loss_calculator.calculate_loss(decoded, observed)

        return loss, (lats, lons, true_lats, true_lons)
