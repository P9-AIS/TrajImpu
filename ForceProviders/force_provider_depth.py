import math

import torch
from ForceTypes.area import Area
from ForceProviders.i_force_provider import IForceProvider
from ForceTypes.tilemap import Tilemap
from ForceUtils.map_transformer import MapTransformerBuilder as MTB
from ForceTypes.params import Params
from ForceTypes.vec3 import Vec3
from dataclasses import dataclass, replace
import os
import numpy as np
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from tqdm import tqdm
from ForceUtils.geo_converter import GeoConverter as gc
from ModelTypes.ais_col_dict import AISColDict


@dataclass
class Config:
    area: Area
    down_scale_factor: int = 1
    output_dir: str = "Outputs/Depth"
    gaussian_sigma: float = 16.0
    low_percentile_cutoff: float = 35.0
    high_percentile_cutoff: float = 99.0
    sensitivity1: float = 2.0
    sensitivity2: float = 6.0


class DepthForceProvider(IForceProvider):
    _vectormap: tuple[np.ndarray, np.ndarray]
    _tilemap: Tilemap
    _cfg: Config
    _data_handler: ForceDataAccessHandlerDb

    def __init__(self, data_handler: ForceDataAccessHandlerDb, cfg: Config):
        self._cfg = cfg
        self._data_handler = data_handler

        self._tilemap = self._handle_get_tilemap()
        self._vectormap = self._get_vectormap(self._tilemap)

    def _handle_get_tilemap(self):
        tilemap_dir = f"{self._cfg.output_dir}/Tilemaps"

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

        tilemap.downscale(self._cfg.down_scale_factor, np.average)
        print(f"Downsampled to {tilemap.get_tile_size()}m, {tilemap.get_dimensions()}")
        tilemap.save(down_scaled_file_name)
        return tilemap

    def _get_tilemap(self):
        tile_size, depth_messages = self._data_handler.get_depths(self._cfg.area)

        print(f"Creating {tile_size}m tile map from {len(depth_messages)} depth messages")

        tilemap = Tilemap(tile_size, self._cfg.area, dtype=np.float32)

        for (E, N, depth) in tqdm(depth_messages, desc="Aggregating tiles"):
            tilemap.update_tile_espg3034(E, N, lambda old_val: depth if old_val == 0 else min(old_val, depth))

        return tilemap

    def _get_vectormap(self, tilemap):
        print(f"Creating vector field from tile map")

        scale = math.sqrt(self._cfg.down_scale_factor)
        scaled_gaussian_sigma = self._cfg.gaussian_sigma / scale

        Z_transformed = (
            MTB(output_dir=f"{self._cfg.output_dir}/Distributions")
            .add_noise(0.01)
            .threshold(0.0, 30)
            # .percentile_threshold(self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity1)
            .capture_distribution("after_power1")
            .gaussian_blur(scaled_gaussian_sigma)
            # .normalize()
            # .power_transform(1 / self._cfg.sensitivity2)
            .build()
        )(tilemap.get_array())

        dz_dy, dz_dx = np.gradient(Z_transformed)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
        vx = -dz_dx / grad_mag * (1-Z_transformed)
        vy = -dz_dy / grad_mag * (1-Z_transformed)

        return vx, vy

    @staticmethod
    def _get_tilemap_file_name(tilemap_dir: str, cfg: Config):
        return (f"{tilemap_dir}/{cfg.area=}-{cfg.down_scale_factor=}.npz")

    def get_vectormap(self) -> tuple[np.ndarray, np.ndarray]:
        return self._vectormap

    def get_force(self, p: Params) -> Vec3:
        x, y = self._tilemap.tile_from_espg3034(*gc.espg4326_to_epsg3034(p.lon, p.lat))

        dim_x, dim_y = self._tilemap.get_dimensions()
        if x < 0 or x >= dim_x or y < 0 or y >= dim_y:
            return Vec3(0.0, 0.0, 0.0)

        x_force = self._vectormap[0][y, x]
        y_force = self._vectormap[1][y, x]

        return Vec3(x_force, y_force, 0.0)

    def get_forces(self, vals: torch.Tensor) -> torch.Tensor:
        # vals: [b*s, num_ais_attr]
        b, s, _ = vals.shape
        forces = []

        for i in range(b):
            batch_forces = []
            for j in range(s):
                lon = vals[i, j, AISColDict.LONGITUDE.value].item()
                lat = vals[i, j, AISColDict.LATITUDE.value].item()
                force_vec = self.get_force(Params(lon=lon, lat=lat))
                batch_forces.append([force_vec.x, force_vec.y, force_vec.z])
            forces.append(batch_forces)

        return torch.tensor(forces, dtype=torch.float32)  # shape [b, s, 3]
