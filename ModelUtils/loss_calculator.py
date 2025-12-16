import torch
from dataclasses import dataclass
from ModelTypes.ais_col_dict import AISColDict


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    pos_distance_loss: torch.Tensor
    delta_hyp_loss: torch.Tensor
    consistency_loss: torch.Tensor
    force_loss: torch.Tensor
    frechet_distance_loss: float

    def __str__(self) -> str:
        return (f"Total Loss: {self.total_loss.item():.4f}\n"
                f"Position Distance Loss: {self.pos_distance_loss.item():.4f}\n"
                f"Delta Hyp Loss: {self.delta_hyp_loss.item():.4f}\n"
                f"Consistency Loss: {self.consistency_loss.item():.4f}\n"
                f"Force Loss: {self.force_loss.item():.4f}\n"
                f"Frechet Distance Loss: {self.frechet_distance_loss:.4f}")


@dataclass
class LossTypes:
    mse: LossOutput
    mae: LossOutput
    smape: LossOutput


class LossCalculator:

    @staticmethod
    def get_loss_type(loss_type: str,
                      full_pos_pred: torch.Tensor, full_pos_true: torch.Tensor,
                      pos_pred: torch.Tensor, pos_true: torch.Tensor,
                      deltas_pred: torch.Tensor, deltas_true: torch.Tensor,
                      total_consistency_loss: torch.Tensor,
                      forces_pred: torch.Tensor, forces_true: torch.Tensor) -> LossOutput:

        pos_distance = LossCalculator._euclidean_distance(pos_pred, pos_true)
        pos_distance_zero = torch.zeros_like(pos_distance)

        pos_distance_loss = LossCalculator.get_loss(
            loss_type,
            pos_distance,
            pos_distance_zero)

        delta_hyp = LossCalculator._calc_hyp(
            deltas_pred[:, :, AISColDict.NORTHERN_DELTA.value],
            deltas_true[:, :, AISColDict.NORTHERN_DELTA.value],
            deltas_pred[:, :, AISColDict.EASTERN_DELTA.value],
            deltas_true[:, :, AISColDict.EASTERN_DELTA.value])
        delta_hyp_zero = torch.zeros_like(delta_hyp)

        delta_hyp_loss = LossCalculator.get_loss(
            loss_type,
            delta_hyp,
            delta_hyp_zero)

        force_loss = LossCalculator.get_loss(loss_type, forces_pred, forces_true)

        total_loss = pos_distance_loss + delta_hyp_loss + 1000 * total_consistency_loss + 10 * force_loss

        frechet_distance_loss = LossCalculator._frechet_distance(full_pos_pred, full_pos_true)

        return LossOutput(
            total_loss=total_loss,
            pos_distance_loss=pos_distance_loss,
            delta_hyp_loss=delta_hyp_loss,
            consistency_loss=total_consistency_loss,
            force_loss=force_loss,
            frechet_distance_loss=frechet_distance_loss
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

    def calculate_loss(self,
                       full_pos_pred: torch.Tensor, full_pos_true: torch.Tensor,
                       pos_pred: torch.Tensor, pos_true: torch.Tensor,
                       deltas_pred: torch.Tensor, deltas_true: torch.Tensor,
                       total_consistency_loss: torch.Tensor,
                       decoded_forces: torch.Tensor, true_forces: torch.Tensor) -> LossTypes:
        return LossTypes(
            mse=self.get_loss_type("mse", full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                   total_consistency_loss, decoded_forces, true_forces),
            mae=self.get_loss_type("mae", full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                   total_consistency_loss, decoded_forces, true_forces),
            smape=self.get_loss_type("smape", full_pos_pred, full_pos_true, pos_pred, pos_true, deltas_pred, deltas_true,
                                     total_consistency_loss, decoded_forces, true_forces)
        )

    @staticmethod
    def _calc_hyp(imputed_lat, ground_truth_lat, imputed_lon, ground_truth_lon) -> torch.Tensor:
        return torch.hypot(imputed_lat - ground_truth_lat, imputed_lon - ground_truth_lon)

    @staticmethod
    def _euclidean_distance(pos_pred, pos_true) -> torch.Tensor:
        lat1 = pos_pred[..., 0]
        lon1 = pos_pred[..., 1]
        lat2 = pos_true[..., 0]
        lon2 = pos_true[..., 1]

        R = 6371000.0  # Earth radius in meters
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        mean_lat = (lat1 + lat2) / 2.0

        x = dlon * torch.cos(mean_lat)
        y = dlat

        return R * torch.sqrt(x * x + y * y)

    @staticmethod
    def _frechet_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
        dist = LossCalculator._euclidean_distance(pred, true)  # (T, T)
        T = dist.shape[0]

        ca = torch.full((T, T), -1.0, device=pred.device)

        def recurse(i, j):
            if ca[i, j] > -0.5:
                return ca[i, j]
            elif i == 0 and j == 0:
                ca[i, j] = dist[0, 0]
            elif i > 0 and j == 0:
                ca[i, j] = torch.max(recurse(i - 1, 0), dist[i, 0])
            elif i == 0 and j > 0:
                ca[i, j] = torch.max(recurse(0, j - 1), dist[0, j])
            elif i > 0 and j > 0:
                ca[i, j] = torch.max(
                    torch.min(torch.stack([
                        recurse(i - 1, j),
                        recurse(i - 1, j - 1),
                        recurse(i, j - 1)
                    ])),
                    dist[i, j]
                )
            else:
                ca[i, j] = float("inf")
            return ca[i, j]

        return recurse(T - 1, T - 1).item()
