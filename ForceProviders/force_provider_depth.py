import math
from Types.area import Area
from ForceProviders.i_force_provider import IForceProvider
from Types.tilemap import Tilemap
from Utils.map_transformer import MapTransformerBuilder as MTB
from Types.params import Params
from Types.vec2 import Vec3
from dataclasses import dataclass, replace
import os
import numpy as np
from DataAccess.data_access_handler import DataAccessHandler
from tqdm import tqdm
from Utils.geo_converter import GeoConverter as gc


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
    _cfg: Config
    _data_handler: DataAccessHandler

    def __init__(self, data_handler: DataAccessHandler, cfg: Config):
        self._cfg = cfg
        self._data_handler = data_handler

        tile_map = self._handle_get_tile_map()
        self._vectormap = self._get_vector_map(tile_map)

    def _handle_get_tile_map(self):
        tile_map_dir = f"{self._cfg.output_dir}/Tilemaps"
        os.makedirs(tile_map_dir, exist_ok=True)

        down_scaled_file_name = self._get_tilemap_file_name(tile_map_dir, self._cfg)
        original_file_name = self._get_tilemap_file_name(tile_map_dir, replace(self._cfg, down_scale_factor=1))

        if os.path.exists(down_scaled_file_name):
            return Tilemap.load(down_scaled_file_name)

        if os.path.exists(original_file_name):
            high_res_tilemap = Tilemap.load(original_file_name)
        else:
            high_res_tilemap = self._get_tile_map()
            Tilemap.save(original_file_name, high_res_tilemap)

        if self._cfg.down_scale_factor == 1:
            return high_res_tilemap

        low_res_tilemap = high_res_tilemap.downscale_tile_map(self._cfg.down_scale_factor, lambda a, b: a + b)
        new_cell_size = 50 * math.sqrt(self._cfg.down_scale_factor)
        print(f"Downsampled to {new_cell_size}m, {len(low_res_tilemap)} unique tiles total")
        Tilemap.save(down_scaled_file_name, low_res_tilemap)
        return low_res_tilemap

    def _get_tile_map(self):
        tile_size, depth_messages = self._data_handler.get_depths(self._cfg.area)

        print(f"Creating {tile_size}m tile map from {len(depth_messages)} depth messages")

        tile_map = Tilemap(tile_size, self._cfg.area)

        for (E, N, depth) in tqdm(depth_messages, desc="Aggregating tiles"):
            tile_map.update_tile_espg3034(E, N, lambda old_val: depth if old_val == 0 else min(old_val, depth))

        return tile_map

    def _get_vector_map(self, tile_map):
        print(f"Creating vector field from tile map")

        dim_x, dim_y = tile_map.get_dimensions()
        Z = np.zeros((dim_y, dim_x), dtype=np.float32)

        for (x, y), count in tqdm(tile_map.items(), total=len(tile_map), desc="Building vector field"):
            Z[y, x] = count

        Z_transformed = (
            MTB(output_dir=f"{self._cfg.output_dir}/Distributions")
            .percentile_threshold(self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity1)
            .capture_distribution("after_power1")
            .add_noise(0.01)
            # .gaussian_blur(self._cfg.gaussian_sigma)
            # .normalize()
            # .power_transform(1 / self._cfg.sensitivity2)
            .build()
        )(Z)

        dz_dy, dz_dx = np.gradient(Z_transformed)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
        vx = -dz_dx / grad_mag * (1-Z_transformed)
        vy = -dz_dy / grad_mag * (1-Z_transformed)

        return vx, vy

    @staticmethod
    def _get_tilemap_file_name(tile_map_dir: str, cfg: Config):
        return (f"{tile_map_dir}/{cfg.area=}-{cfg.down_scale_factor=}.pkl")

    def get_force(self, p: Params) -> Vec3:
        x, y = self._tile_map.tile_from_espg3034(*gc.espg4326_to_epsg3034(p.lon, p.lat))
        x_force = self._vectormap[0][y, x]
        y_force = self._vectormap[1][y, x]

        return Vec3(x_force, y_force, 0.0)
