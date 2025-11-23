from io import BytesIO

from matplotlib import image
from ForceData.i_force_data_upload_handler import IForceDataUploadHandler
from ForceTypes.area import Area
from ModelTypes.ais_dataset_masked import AISDatasetMasked
import requests
import gzip
import json
from dataclasses import dataclass
from PIL import Image
from ForceUtils.geo_converter import GeoConverter as GC


@dataclass
class Config:
    server_address: str = "http://localhost:4000"


class ForceDataUploadHandlerHTTP(IForceDataUploadHandler):
    def __init__(self, config: Config):
        self._cfg = config

    def upload_traj(self, dataset: AISDatasetMasked, start_idx: int, end_idx: int) -> None:
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

    def upload_image(self, image_path: str, name: str, area_3034: Area) -> None:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        ext = image_path.split(".")[-1].lower()
        image_format = ext if ext in ["png", "jpg", "jpeg", "tif", "tiff"] else "png"

        top_right_lon, top_right_lat = GC.epsg3034_to_espg4326(area_3034.top_right.E, area_3034.top_right.N)
        bottom_left_lon, bottom_left_lat = GC.epsg3034_to_espg4326(area_3034.bottom_left.E, area_3034.bottom_left.N)

        json_area = json.dumps({
            "top_right": {"lat": top_right_lat, "lon": top_right_lon},
            "bottom_left": {"lat": bottom_left_lat, "lon": bottom_left_lon}
        })

        files = {
            "image": (f"{name}.{image_format.lower()}", img_bytes, f"image/{image_format.lower()}"),
        }
        data = {
            "name": name,
            "area": json_area
        }

        print(f"Uploading image '{name}' to {self._cfg.server_address}...")
        response = requests.post(
            f"{self._cfg.server_address}/image",
            files=files,
            data=data
        )

        print(response.status_code, response.json())
