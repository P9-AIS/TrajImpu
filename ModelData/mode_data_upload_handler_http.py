from ModelData.i_model_data_upload_handler import IModelDataUploadHandler
from ModelTypes.ais_dataset_masked import AISDatasetMasked
import numpy as np
import requests
import gzip
import json
from dataclasses import dataclass


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

        data = {"trajectory": dataset.data[start_idx:end_idx].tolist()}

        json_bytes = json.dumps(data).encode("utf-8")
        compressed = gzip.compress(json_bytes)

        print(f"Uploading trajectories {start_idx} to {end_idx} to {self._cfg.server_address}...")
        response = requests.post(
            f"{self._cfg.server_address}/trajectory",
            data=compressed,
            headers={"Content-Type": "application/octet-stream"}  # just raw bytes
        )

        print(response.status_code, response.json())
