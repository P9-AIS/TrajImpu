import torch
from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelTypes.ais_dataset_masked import AISDatasetMasked
import numpy as np
import requests
import gzip
import json
from dataclasses import dataclass
from ForceUtils.geo_converter import GeoConverter as gc


@dataclass
class Config:
    server_address: str = "http://localhost:4000"


class ModelDataUploadHandlerHTTP(IModelDataUploadHandler):
    def __init__(self, config: Config):
        self._cfg = config

    def upload_trajectories(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
        end_idx = min(end_idx, len(dataset))

        if end_idx == -1:
            end_idx = len(dataset)

        northerns = dataset.northerns[start_idx:end_idx]
        easterns = dataset.easterns[start_idx:end_idx]

        lons, lats = gc.espg3034_to_epsg4326_batch(easterns, northerns)
        data = np.stack((lats, lons), axis=-1)

        data = {"trajectory": data.tolist()}

        json_bytes = json.dumps(data).encode("utf-8")
        compressed = gzip.compress(json_bytes)

        print(f"Uploading trajectories {start_idx} to {end_idx} to {self._cfg.server_address}...")
        response = requests.post(
            f"{self._cfg.server_address}/trajectories",
            data=compressed,
            headers={"Content-Type": "application/octet-stream"}  # just raw bytes
        )

        print(response.status_code, response.json(), "\n")

    def upload_predictions(self, model_name, masks, predicted_northerns: torch.Tensor, predicted_easterns: torch.Tensor,
                           true_northerns: torch.Tensor, true_easterns: torch.Tensor) -> None:

        masks_new = masks[..., 0]

        pred_lons, pred_lats = gc.epsg3034_to_espg4326_batch_tensor(predicted_easterns, predicted_northerns)
        true_lons, true_lats = gc.epsg3034_to_espg4326_batch_tensor(true_easterns, true_northerns)

        concat = torch.stack([
            masks_new,
            pred_lats,
            pred_lons,
            true_lats,
            true_lons
        ], dim=-1)

        concat = concat.cpu().numpy()

        data = {"predictions": concat.tolist()}

        json_bytes = json.dumps(data).encode("utf-8")
        compressed = gzip.compress(json_bytes)

        print(f"Uploading predictions to {self._cfg.server_address}...")
        response = requests.post(
            f"{self._cfg.server_address}/predictions/{model_name}",
            data=compressed,
            headers={"Content-Type": "application/octet-stream"}  # just raw bytes
        )

        print(response.status_code, response.json(), "\n")

    def reset_predictions(self, model_name) -> None:
        print(f"Resetting predictions for model {model_name} on server at {self._cfg.server_address}...")
        response = requests.post(f"{self._cfg.server_address}/predictions/{model_name}/reset")

        print(response.status_code, response.json(), "\n")
