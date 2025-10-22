import math
from Types.area import Area
from ForceProviders.i_force_provider import IForceProvider
from Types.tilemap import Tilemap
from params import Params
from Types.vec2 import Vec2
from dataclasses import dataclass, field
from vessel_types import VesselType
import datetime as dt
import pickle
import os
import numpy as np
from DataAccess.data_access_handler import DataAccessHandler
from tqdm import tqdm
from Utils.map_transformer import MapTransformerBuilder as MTB


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
    _vectormap: tuple[Tilemap[float], Tilemap[float]]
    _cfg: Config
    _data_handler: DataAccessHandler

    def __init__(self, data_handler: DataAccessHandler, cfg: Config):
        self._cfg = cfg
        self._data_handler = data_handler
        tile_map_dir = f"{cfg.output_dir}/Tilemaps"
        file_name = self._get_tilemap_file_name(tile_map_dir, cfg)
        os.makedirs(tile_map_dir, exist_ok=True)

        if os.path.exists(file_name):
            print(f"Loading tile map from '{file_name}'")
            with open(file_name, 'rb') as f:
                tile_map: Tilemap = pickle.load(f)
            print(f"Loaded {len(tile_map)} tiles from '{file_name}'\n")
        else:
            high_res_tiles = self._get_tile_map()
            print(f"Saving tile map to '{file_name}'")
            with open(file_name, 'wb') as f:
                pickle.dump(high_res_tiles, f)
            print(f"Saved {len(high_res_tiles)} tiles to '{file_name}'\n")
            tile_map: Tilemap = high_res_tiles

        tile_map = tile_map.downscale_tile_map(self._cfg.down_scale_factor, lambda a, b: a + b)

        print(
            f"Downsampled tile map to {self._cfg.base_tile_size_m * math.sqrt(self._cfg.down_scale_factor)}m, now contains {len(tile_map)} tiles")

        self._vectormap = self._get_vector_map(tile_map)

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

        Z_transformed = (
            MTB(output_dir=f"{self._cfg.output_dir}/Distributions")
            .percentile_threshold(self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity1)
            .capture_distribution("after_power1")
            .add_noise(0.01)
            .sato_filter(self._cfg.sato_sigmas)
            .normalize()
            .gaussian_blur(self._cfg.gaussian_sigma)
            .normalize()
            .power_transform(1 / self._cfg.sensitivity2)
            .build()
        )(Z)

        dz_dy, dz_dx = np.gradient(Z_transformed)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
        vx = -dz_dx / grad_mag * (1 - Z_transformed)
        vy = -dz_dy / grad_mag * (1 - Z_transformed)

        return (
            Tilemap.from_2d(vx, tile_map.get_tile_size(), tile_map.get_espg3034_bounds()),
            Tilemap.from_2d(vy, tile_map.get_tile_size(), tile_map.get_espg3034_bounds())
        )

    @staticmethod
    def _get_tilemap_file_name(tile_map_dir: str, cfg: Config):
        return (f"{tile_map_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.sample_rate=}-{cfg.base_tile_size_m=}.pkl")

    def get_force(self, p: Params) -> Vec2:
        x, y = self._vector_map[0].tile_from_espg4326(p.lon, p.lat)

        x_force = self._vector_map[0][x, y]
        y_force = self._vector_map[1][x, y]

        return Vec2(x_force, y_force)
