from DataAccess.i_data_access_handler import AreaTuple
from ForceProviders.i_force_provider import IForceProvider
from Types.latlon import LatLon
from params import Params
from Types.vec2 import Vec2, create_vec2_array
from dataclasses import dataclass
from vessel_types import VesselType
import datetime as dt
import mercantile
from collections import Counter
import pickle
import os
import numpy as np
from DataAccess.data_access_handler import DataAccessHandler


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    sample_rate: int
    area: AreaTuple
    vessel_types: list[VesselType]
    base_zoom: int
    active_zoom: int
    output_dir: str = "Outputs/Tilemaps"


class TrafficForceProvider(IForceProvider):
    _vector_map: list[list[Vec2]]
    _cfg: Config
    _data_handler: DataAccessHandler

    def __init__(self, cfg: Config, data_handler: DataAccessHandler):
        self._cfg = cfg
        self._data_handler = data_handler
        file_name = self._get_tilemap_file_name(cfg)
        os.makedirs(cfg.output_dir, exist_ok=True)
        tile_map = 0

        if os.path.exists(file_name):
            print(f"Loading tile map from '{file_name}'")
            with open(file_name, 'rb') as f:
                tile_map = pickle.load(f)
            print(f"Loaded {len(tile_map)} tiles from '{file_name}'\n")
        else:
            high_res_tiles = self._get_tiles()
            print(f"Saving tile map to '{file_name}'")
            with open(file_name, 'wb') as f:
                pickle.dump(high_res_tiles, f)
            print(f"Saved {len(high_res_tiles)} tiles to '{file_name}'\n")
            tile_map = high_res_tiles

        tile_map = self._get_tile_map_at_zoom(tile_map, cfg.active_zoom)
        self._vector_map = self._get_vector_field(tile_map)

    def _get_tiles(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date < self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.sample_rate)

        ais_messages = self._data_handler.get_ais_messages(days, self._cfg.area)

        print(f"Pre-computing tiles at base zoom {self._cfg.base_zoom}")
        high_res_tiles = []
        for msg in ais_messages:
            tile = mercantile.tile(msg.longitude, msg.latitude, zoom=self._cfg.base_zoom)
            high_res_tiles.append(tile)

        return high_res_tiles

    @staticmethod
    def _get_tile_map_at_zoom(tile_map, zoom: int):
        if not tile_map:
            return Counter()

        print(f"Downscaling tiles to zoom {zoom}")

        base_zoom = tile_map[0].z
        if zoom > base_zoom:
            raise ValueError(
                f"Target zoom ({zoom}) cannot be greater than the base zoom ({base_zoom}).")

        parent_tiles = (mercantile.parent(tile, zoom=zoom) for tile in tile_map)

        return Counter(parent_tiles)

    def _get_vector_field(self, tile_map):
        print(f"Creating vector field from tile map")

        bot_left_tile = mercantile.tile(self._cfg.area.bot_left.lon,
                                        self._cfg.area.bot_left.lat, zoom=self._cfg.active_zoom)
        top_right_tile = mercantile.tile(self._cfg.area.top_right.lon,
                                         self._cfg.area.top_right.lat, zoom=self._cfg.active_zoom)

        num_x_tiles = top_right_tile.x - bot_left_tile.x
        num_y_tiles = bot_left_tile.y - top_right_tile.y

        Z = np.zeros((num_y_tiles, num_x_tiles), dtype=np.float32)

        for tile, count in tile_map.items():
            if (bot_left_tile.x <= tile.x < top_right_tile.x and
                    top_right_tile.y <= tile.y < bot_left_tile.y):

                idx_x = tile.x - bot_left_tile.x
                idx_y = tile.y - top_right_tile.y

                if 0 <= idx_y < num_y_tiles and 0 <= idx_x < num_x_tiles:
                    Z[idx_y, idx_x] = count

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
        return (f"{cfg.output_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.area=}-{cfg.sample_rate=}-{cfg.base_zoom=}.pkl")

    def get_force(self, p: Params) -> Vec2:
        top_left_tile = mercantile.tile(self._cfg.area_top_left_lon,
                                        self._cfg.area_top_left_lat, zoom=self._cfg.active_zoom)

        active_tile = mercantile.tile(p.lon, p.lat, zoom=self._cfg.active_zoom)

        idx_x = active_tile.x - top_left_tile.x
        idx_y = active_tile.y - top_left_tile.y

        return self._vector_map[idx_y][idx_x]
