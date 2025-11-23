from abc import ABC, abstractmethod
import torch
from ModelTypes.ais_dataset_masked import AISDatasetMasked


class IModelDataUploadHandler(ABC):
    @abstractmethod
    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        pass

    @abstractmethod
    def upload_predictions(self, step: int, predicted_lats: torch.Tensor, predicted_lons: torch.Tensor,
                           true_lats: torch.Tensor, true_lons: torch.Tensor) -> None:
        pass

    @abstractmethod
    def reset_predictions(self) -> None:
        pass
