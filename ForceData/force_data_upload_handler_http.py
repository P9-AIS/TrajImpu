from io import BytesIO
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

    def upload_image(self, image: Image.Image, name: str, area: Area) -> None:
        buf = BytesIO()
        image_format = image.format if image.format else "PNG"
        image.save(buf, format=image_format)
        buf.seek(0)

        top_right_lon, top_right_lat = GC.epsg3034_to_espg4326(area.top_right.E, area.top_right.N)
        bottom_left_lon, bottom_left_lat = GC.epsg3034_to_espg4326(area.bottom_left.E, area.bottom_left.N)

        json_area = json.dumps({
            "top_right": {"lat": top_right_lat, "lon": top_right_lon},
            "bottom_left": {"lat": bottom_left_lat, "lon": bottom_left_lon}
        })

        files = {
            "image": (f"{name}.{image_format.lower()}", buf, f"image/{image_format.lower()}"),
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
