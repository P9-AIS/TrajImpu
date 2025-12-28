from abc import ABC, abstractmethod
import torch
from ModelTypes.ais_dataset_masked import AISDatasetMasked


class IModelDataUploadHandler(ABC):
    @abstractmethod
    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        pass

    @abstractmethod
    def upload_predictions(self, model_name, masks, predicted_northerns: torch.Tensor, predicted_easterns: torch.Tensor,
                           true_northerns: torch.Tensor, true_easterns: torch.Tensor) -> None:
        pass

    @abstractmethod
    def reset_predictions(self, model_name) -> None:
        pass
