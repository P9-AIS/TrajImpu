import math
from Types.area import Area
from ForceProviders.i_force_provider import IForceProvider
from Types.tilemap import Tilemap
from Types.params import Params
from Types.vec2 import Vec3
from dataclasses import dataclass, field, replace
from Types.vessel_types import VesselType
import datetime as dt
import os
import numpy as np
from DataAccess.data_access_handler import DataAccessHandler
from tqdm import tqdm
from Utils.map_transformer import MapTransformerBuilder as MTB
from Utils.geo_converter import GeoConverter as gc


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    sample_rate: int
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
    _data_handler: DataAccessHandler
    _tile_map: Tilemap

    def __init__(self, data_handler: DataAccessHandler, cfg: Config):
        self._cfg = cfg
        self._data_handler = data_handler

        self._tile_map = self._handle_get_tile_map()
        self._vectormap = self._get_vector_map(self._tile_map)

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
        new_cell_size = self._cfg.base_tile_size_m * math.sqrt(self._cfg.down_scale_factor)
        print(f"Downsampled to {new_cell_size}m, {len(low_res_tilemap)} unique tiles total")
        Tilemap.save(down_scaled_file_name, low_res_tilemap)
        return low_res_tilemap

    def _get_tile_map(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date <= self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.sample_rate)

        ais_messages = self._data_handler.get_ais_messages_no_stops(days, self._cfg.area)

        print(f"Creating {self._cfg.base_tile_size_m}m tile map from {len(ais_messages)} AIS messages")

        tile_map = Tilemap(self._cfg.base_tile_size_m, self._cfg.area)

        for (lon, lat) in tqdm(ais_messages, desc="Aggregating tiles"):
            tile_map.update_tile_espg4326(lon, lat, lambda old_val: old_val + 1)

        return tile_map

    def _get_vector_map(self, tile_map):
        print(f"Creating vector field from tile map")

        dim_x, dim_y = tile_map.get_dimensions()
        Z = np.zeros((dim_y, dim_x), dtype=np.float32)

        for (x, y), count in tqdm(tile_map.items(), total=len(tile_map), desc="Building vector field"):
            Z[y, x] = count

        scale = math.sqrt(self._cfg.down_scale_factor)
        scaled_sato_sigmas = [s / scale for s in self._cfg.sato_sigmas]
        scaled_gaussian_sigma = self._cfg.gaussian_sigma / scale

        Z_transformed = (
            MTB(output_dir=f"{self._cfg.output_dir}/Distributions")
            .percentile_threshold(self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity1)
            .capture_distribution("after_power1")
            .add_noise(1e-8)
            .sato_filter(scaled_sato_sigmas)
            .normalize()
            .gaussian_blur(scaled_gaussian_sigma)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity2)
            .build()
        )(Z)

        dz_dy, dz_dx = np.gradient(Z_transformed)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
        vx = -dz_dx / grad_mag * (1 - Z_transformed)
        vy = -dz_dy / grad_mag * (1 - Z_transformed)

        return vx, vy

    @staticmethod
    def _get_tilemap_file_name(tile_map_dir: str, cfg: Config):
        return (f"{tile_map_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.sample_rate=}-{cfg.base_tile_size_m=}-{cfg.down_scale_factor=}.pkl")

    def get_vector_map(self) -> tuple[np.ndarray, np.ndarray]:
        return self._vectormap

    def get_force(self, p: Params) -> Vec3:
        x, y = self._tile_map.tile_from_espg3034(*gc.espg4326_to_epsg3034(p.lon, p.lat))
        x_force = self._vectormap[0][y, x]
        y_force = self._vectormap[1][y, x]

        return Vec3(x_force, y_force, 0.0)
