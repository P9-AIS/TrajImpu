import torch
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelTypes.ais_dataset_masked import AISDatasetMasked


class ModelDataUploadHandlerMock(IModelDataUploadHandler):
    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        pass

    def upload_predictions(self, model_name, masks, predicted_lats: torch.Tensor, predicted_lons: torch.Tensor,
                           true_lats: torch.Tensor, true_lons: torch.Tensor) -> None:
        pass

    def reset_predictions(self, model_name) -> None:
        pass
