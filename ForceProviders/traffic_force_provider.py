from ForceProviders.i_force_provider import IForceProvider
from params import Params
from vec2 import Vec2, create_vec2_array
from dataclasses import dataclass
from vessel_types import VesselType
import datetime as dt
import mercantile
from collections import Counter
import pickle
import os
import numpy as np


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    sample_rate: int
    area_top_left_lat: float
    area_top_left_lon: float
    area_bottom_right_lat: float
    area_bottom_right_lon: float
    vessel_types: list[VesselType]
    base_zoom: int
    active_zoom: int
    output_dir: str


class TrafficForceProvider(IForceProvider):
    _vector_map: list[list[Vec2]]
    _cfg: Config

    def __init__(self, cfg: Config):
        self._cfg = cfg
        file_name = self._get_tilemap_file_name(cfg)
        os.makedirs(cfg.output_dir, exist_ok=True)
        tile_map = 0

        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                tile_map = pickle.load(f)
        else:
            high_res_tiles = self._get_tile_map()
            with open(file_name, 'wb') as f:
                pickle.dump(high_res_tiles, f)
            print(f"Saved {len(high_res_tiles)} tiles to '{file_name}'\n")
            tile_map = high_res_tiles

        tile_map = self._get_tile_map_at_zoom(tile_map, cfg.active_zoom)
        self._vector_map = self._get_vector_field(tile_map)

    def _get_tile_map(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date < self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.sample_rate)

        ais_messages = self._get_ais_messages(days)

        print(f"--- Step 1: Pre-computing tiles at base zoom {self._cfg.base_zoom} ---")
        high_res_tiles = []
        for msg in ais_messages:
            tile = mercantile.tile(msg['lon'], msg['lat'], zoom=self._cfg.base_zoom)
            high_res_tiles.append(tile)

        return high_res_tiles

    @staticmethod
    def _get_tile_map_at_zoom(tile_map, zoom: int):
        if not tile_map:
            return Counter()

        base_zoom = tile_map[0].z
        if zoom > base_zoom:
            raise ValueError(
                f"Target zoom ({zoom}) cannot be greater than the base zoom ({base_zoom}).")

        parent_tiles = (mercantile.parent(tile, zoom=zoom)
                        for tile in tile_map)

        return Counter(parent_tiles)

    @staticmethod
    def _get_ais_messages(days: list[dt.date]):
        ais_messages = [
            {'mmsi': '219000001', 'lat': 57.04, 'lon': 9.92},
            {'mmsi': '219000002', 'lat': 57.05, 'lon': 9.93},
            {'mmsi': '219000002', 'lat': 57.066227, 'lon': 9.790968},
            {'mmsi': '367000001', 'lat': 40.71, 'lon': -74.00},
            {'mmsi': '367000002', 'lat': 40.72, 'lon': -74.01},
        ]
        return ais_messages

    def _get_vector_field(self, tile_map):
        top_left_tile = mercantile.tile(self._cfg.area_top_left_lon,
                                        self._cfg.area_top_left_lat, zoom=self._cfg.active_zoom)
        bottom_right_tile = mercantile.tile(self._cfg.area_bottom_right_lon,
                                            self._cfg.area_bottom_right_lat, zoom=self._cfg.active_zoom)

        num_x_tiles = bottom_right_tile.x - top_left_tile.x
        num_y_tiles = bottom_right_tile.y - top_left_tile.y

        tile_counts = []
        for y in range(num_y_tiles):
            row_counts = []
            for x in range(num_x_tiles):
                tile = mercantile.Tile(top_left_tile.x + x, top_left_tile.y + y, self._cfg.active_zoom)
                count = tile_map.get(tile, 0)
                row_counts.append(count)
            tile_counts.append(row_counts)

        Z = np.array(tile_counts, dtype=np.float32)
        y = np.arange(Z.shape[0])
        x = np.arange(Z.shape[1])
        dz_dy, dz_dx = np.gradient(Z)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
        vx = -dz_dx / (grad_mag + 1e-8)
        vy = -dz_dy / (grad_mag + 1e-8)
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
        vx *= Z_norm
        vy *= Z_norm

        return create_vec2_array(vx, vy)

    @staticmethod
    def _get_tilemap_file_name(cfg: Config):
        return (f"{cfg.output_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.sample_rate=}-{cfg.base_zoom=}.pkl")

    def get_force(self, p: Params) -> Vec2:
        top_left_tile = mercantile.tile(self._cfg.area_top_left_lon,
                                        self._cfg.area_top_left_lat, zoom=self._cfg.active_zoom)

        active_tile = mercantile.tile(p.lon, p.lat, zoom=self._cfg.active_zoom)

        idx_x = active_tile.x - top_left_tile.x
        idx_y = active_tile.y - top_left_tile.y

        return self._vector_map[idx_y][idx_x]
