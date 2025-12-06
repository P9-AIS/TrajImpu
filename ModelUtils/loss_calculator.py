import torch
from dataclasses import dataclass
from ModelTypes.ais_col_dict import AISColDict


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    lat_loss: torch.Tensor
    lon_loss: torch.Tensor
    haversine_loss: torch.Tensor
    consistency_loss: torch.Tensor
    force_loss: torch.Tensor

    def __str__(self) -> str:
        return (f"Total Loss: {self.total_loss.item():.4f}\n"
                f"Lat Loss: {self.lat_loss.item():.4f}\n"
                f"Lon Loss: {self.lon_loss.item():.4f}\n"
                f"Haversine Loss: {self.haversine_loss.item():.4f}\n"
                f"Consistency Loss: {self.consistency_loss.item():.4f}")


@dataclass
class LossTypes:
    mse: LossOutput
    mae: LossOutput
    smape: LossOutput


class LossCalculator:

    @staticmethod
    def get_loss_type(loss_type: str, prediction, truth, total_consistency_loss: torch.Tensor, decoded_forces: torch.Tensor, true_forces: torch.Tensor) -> LossOutput:

        lat_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.NORTHERN_DELTA.value:AISColDict.NORTHERN_DELTA.value+1],
            truth[:, :, AISColDict.NORTHERN_DELTA.value:AISColDict.NORTHERN_DELTA.value+1])

        lon_loss = LossCalculator.get_loss(
            loss_type,
            prediction[:, :, AISColDict.EASTERN_DELTA.value:AISColDict.EASTERN_DELTA.value+1],
            truth[:, :, AISColDict.EASTERN_DELTA.value:AISColDict.EASTERN_DELTA.value+1])

        haversine_loss = LossCalculator._calc_spatial_loss(
            prediction[:, :, AISColDict.NORTHERN_DELTA.value],
            truth[:, :, AISColDict.NORTHERN_DELTA.value],
            prediction[:, :, AISColDict.EASTERN_DELTA.value],
            truth[:, :, AISColDict.EASTERN_DELTA.value])

        force_loss = LossCalculator.get_loss(
            loss_type,
            decoded_forces,
            true_forces)

        total_loss = lat_loss + lon_loss + 1000 * total_consistency_loss + 10 * force_loss

        return LossOutput(
            total_loss=total_loss,
            lat_loss=lat_loss,
            lon_loss=lon_loss,
            haversine_loss=haversine_loss,
            consistency_loss=total_consistency_loss,
            force_loss=force_loss
        )

    @staticmethod
    def get_loss(loss_func: str, prediction, truth) -> torch.Tensor:
        if loss_func == "mse":
            return torch.nn.functional.mse_loss(prediction, truth)
        elif loss_func == "mae":
            return torch.nn.functional.l1_loss(prediction, truth)
        elif loss_func == "smape":
            # SMAPE formula: 2 * |y_pred - y_true| / (|y_pred| + |y_true|)
            numerator = torch.abs(prediction - truth)
            denominator = torch.abs(prediction) + torch.abs(truth)

            epsilon = 1e-8
            return 2 * torch.mean(numerator / (denominator + epsilon))
        else:
            raise ValueError(f"Unsupported loss function: {loss_func}")

    def calculate_loss(self, imputed: torch.Tensor, ground_truth: torch.Tensor, total_consistency_loss: torch.Tensor, decoded_forces: torch.Tensor, true_forces: torch.Tensor) -> LossTypes:
        return LossTypes(
            mse=self.get_loss_type("mse", imputed, ground_truth, total_consistency_loss, decoded_forces, true_forces),
            mae=self.get_loss_type("mae", imputed, ground_truth, total_consistency_loss, decoded_forces, true_forces),
            smape=self.get_loss_type("smape", imputed, ground_truth,
                                     total_consistency_loss, decoded_forces, true_forces)
        )

    @staticmethod
    def _calc_spatial_loss(imputed_lat, ground_truth_lat, imputed_lon, ground_truth_lon) -> torch.Tensor:
        dist = torch.hypot(imputed_lat - ground_truth_lat, imputed_lon - ground_truth_lon)
        return torch.mean(dist)
