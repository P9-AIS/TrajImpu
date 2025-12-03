import torch
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelTypes.ais_dataset_masked import AISDatasetMasked


class ModelDataUploadHandlerMock(IModelDataUploadHandler):
    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        print(f"Mock upload of trajectories {start_idx} to {end_idx} to mock server...")

    def upload_predictions(self, step: int, predicted_lats: torch.Tensor, predicted_lons: torch.Tensor,
                           true_lats: torch.Tensor, true_lons: torch.Tensor) -> None:
        print(f"Mock upload of predictions at step {step} to mock server...")

    def reset_predictions(self) -> None:
        print(f"Mock reset of predictions on mock server...")
