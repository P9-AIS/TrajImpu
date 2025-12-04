import math

import torch
from ForceTypes.area import Area
from ForceProviders.i_force_provider import IForceProvider
from ForceTypes.tilemap import Tilemap
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3
from dataclasses import dataclass, field, replace
from ForceTypes.vessel_types import VesselType
import datetime as dt
import os
import numpy as np
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from tqdm import tqdm
from ForceUtils.map_transformer import MapTransformerBuilder as MTB
from ForceUtils.geo_converter import GeoConverter as gc
from ModelTypes.ais_col_dict import AISColDict


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    date_step: int
    area: Area
    vessel_types: list[VesselType]
    base_tile_size_m: int = 50
    down_scale_factor: int = 1
    output_dir: str = "Outputs/Traffic"
    sato_sigmas: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    gaussian_sigma: float = 16.0
    low_percentile_cutoff: float = 35.0
    high_percentile_cutoff: float = 99.0
    sensitivity1: float = 2.0
    sensitivity2: float = 6.0


class TrafficForceProvider(IForceProvider):
    _vectormap: tuple[np.ndarray, np.ndarray]
    _cfg: Config
    _data_handler: ForceDataAccessHandlerDb
    _tilemap: Tilemap

    def __init__(self, data_handler: ForceDataAccessHandlerDb, cfg: Config):
        self._cfg = cfg
        self._data_handler = data_handler

        self._tilemap = self._handle_get_tilemap()
        self._vectormap = self._get_vectormap(self._tilemap)

    def _handle_get_tilemap(self):
        tilemap_dir = f"{self._cfg.output_dir}/Tilemaps"
        os.makedirs(tilemap_dir, exist_ok=True)

        down_scaled_file_name = self._get_tilemap_file_name(tilemap_dir, self._cfg)
        original_file_name = self._get_tilemap_file_name(tilemap_dir, replace(self._cfg, down_scale_factor=1))

        if os.path.exists(down_scaled_file_name):
            return Tilemap.load(down_scaled_file_name)

        if os.path.exists(original_file_name):
            tilemap = Tilemap.load(original_file_name)
        else:
            tilemap = self._get_tilemap()
            tilemap.save(original_file_name)

        if self._cfg.down_scale_factor == 1:
            return tilemap

        tilemap.downscale(self._cfg.down_scale_factor, np.sum)
        print(f"Downsampled to {tilemap.get_tile_size()}m, {tilemap.get_dimensions()}")
        tilemap.save(down_scaled_file_name)
        return tilemap

    def _get_tilemap(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date <= self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.date_step)

        ais_messages = self._data_handler.get_ais_messages_no_stops(days, self._cfg.area)

        print(f"Creating {self._cfg.base_tile_size_m}m tile map from {len(ais_messages)} AIS messages")

        tilemap = Tilemap(self._cfg.base_tile_size_m, self._cfg.area, dtype=np.int32)

        for (_, lon, lat, _, _, _) in tqdm(ais_messages, desc="Aggregating tiles"):
            tilemap.update_tile_espg4326(lon, lat, lambda old_val: old_val + 1)

        return tilemap

    def _get_vectormap(self, tilemap):
        print(f"Creating vector field from tile map")

        scale = math.sqrt(self._cfg.down_scale_factor)
        scaled_sato_sigmas = [s / scale for s in self._cfg.sato_sigmas]
        scaled_gaussian_sigma = self._cfg.gaussian_sigma / scale

        Z_transformed = (
            MTB(output_dir=f"{self._cfg.output_dir}/Distributions")
            .percentile_threshold(self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity1)
            .add_noise(0.01)
            .capture_distribution("after_power1")
            # .sato_filter(scaled_sato_sigmas)
            # .normalize()
            .gaussian_blur(scaled_gaussian_sigma)
            .normalize()
            # .power_transform(1 / self._cfg.sensitivity2)
            .build()
        )(tilemap.get_array())

        dy, dx = np.gradient(Z_transformed)
        grad_mag = np.sqrt(dx**2 + dy**2) + 1e-8
        vx = dx / grad_mag * (1 - Z_transformed)
        vy = dy / grad_mag * (1 - Z_transformed)

        return vx, vy

    @staticmethod
    def _get_tilemap_file_name(tilemap_dir: str, cfg: Config):
        return (f"{tilemap_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.date_step=}-{cfg.base_tile_size_m=}-{cfg.down_scale_factor=}.npz")

    def get_vectormap(self) -> tuple[np.ndarray, np.ndarray]:
        return self._vectormap

    def get_force(self, p: Params) -> Vec3:
        x, y = self._tilemap.tile_from_espg3034(*gc.espg4326_to_epsg3034(p.lon, p.lat))
        x_force = self._vectormap[0][y, x]
        y_force = self._vectormap[1][y, x]

        return Vec3(x_force, y_force, 0.0)

    def get_forces(self, vals: torch.Tensor) -> torch.Tensor:
        # vals: [b, s, num_ais_attr]
        b, s, _ = vals.shape
        forces = []

        for i in range(b):
            batch_forces = []
            for j in range(s):
                lon = vals[i, j, AISColDict.EASTERN_DELTA.value].item()
                lat = vals[i, j, AISColDict.NORTHERN_DELTA.value].item()
                force_vec = self.get_force(Params(lon=lon, lat=lat))
                batch_forces.append([force_vec.x, force_vec.y, force_vec.z])
            forces.append(batch_forces)

        return torch.tensor(forces, dtype=torch.float32)  # shape [b, s, 3]
