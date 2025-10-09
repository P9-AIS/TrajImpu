from DataAccess.i_data_access_handler import AreaTuple
from ForceProviders.i_force_provider import IForceProvider
from Types.tilemap import Tilemap
from params import Params
from Types.vec2 import Vec2
from dataclasses import dataclass
from vessel_types import VesselType
import datetime as dt
import mercantile
import pickle
import os
import numpy as np
from DataAccess.data_access_handler import DataAccessHandler
from tqdm import tqdm


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    sample_rate: int
    area: AreaTuple
    vessel_types: list[VesselType]
    base_zoom: int
    target_zoom: int
    output_dir: str = "Outputs/Tilemaps"


class TrafficForceProvider(IForceProvider):
    _vectormap: tuple[Tilemap[float], Tilemap[float]]
    _cfg: Config
    _data_handler: DataAccessHandler

    def __init__(self, cfg: Config, data_handler: DataAccessHandler):
        self._cfg = cfg
        self._data_handler = data_handler
        file_name = self._get_tilemap_file_name(cfg)
        os.makedirs(cfg.output_dir, exist_ok=True)

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

        tile_map = tile_map.downscale_tile_map(self._cfg.target_zoom)
        print(f"Downsampled tile map to zoom {self._cfg.target_zoom}, now contains {len(tile_map)} tiles")

        self._vectormap = self._get_vector_map(tile_map)

    def _get_tile_map(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date < self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.sample_rate)

        ais_messages = self._data_handler.get_ais_messages(days, self._cfg.area)

        print(f"Pre-computing tiles counts at base zoom {self._cfg.base_zoom}")

        bot_left_tile = mercantile.tile(self._cfg.area.bot_left.lon,
                                        self._cfg.area.bot_left.lat, zoom=self._cfg.base_zoom)
        top_right_tile = mercantile.tile(self._cfg.area.top_right.lon,
                                         self._cfg.area.top_right.lat, zoom=self._cfg.base_zoom)

        tile_map = Tilemap(self._cfg.base_zoom,
                           min_x_tile=bot_left_tile.x, max_x_tile=top_right_tile.x,
                           min_y_tile=top_right_tile.y, max_y_tile=bot_left_tile.y)

        for msg in tqdm(ais_messages, desc="Aggregating tiles"):
            tile = mercantile.tile(msg.longitude, msg.latitude, zoom=self._cfg.base_zoom)
            tile_map.increment(tile.x, tile.y)

        print(f"Computed {len(tile_map)} unique tiles at zoom {self._cfg.base_zoom}")

        return tile_map

    def _get_vector_map(self, tile_map):
        print(f"Creating vector field from tile map")

        bot_left_tile = mercantile.tile(self._cfg.area.bot_left.lon,
                                        self._cfg.area.bot_left.lat, zoom=self._cfg.target_zoom)
        top_right_tile = mercantile.tile(self._cfg.area.top_right.lon,
                                         self._cfg.area.top_right.lat, zoom=self._cfg.target_zoom)

        num_x_tiles = top_right_tile.x - bot_left_tile.x + 1
        num_y_tiles = bot_left_tile.y - top_right_tile.y + 1

        Z = np.zeros((num_y_tiles, num_x_tiles), dtype=np.float32)

        for (x, y), count in tqdm(tile_map.items(), total=len(tile_map), desc="Building vector field"):
            idx_x = x - bot_left_tile.x
            idx_y = y - top_right_tile.y
            Z[idx_y, idx_x] = count

        dz_dy, dz_dx = np.gradient(Z)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
        vx = -dz_dx / (grad_mag + 1e-8)
        vy = -dz_dy / (grad_mag + 1e-8)
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
        vx *= Z_norm
        vy *= Z_norm

        return (Tilemap.from_2d(vx, self._cfg.target_zoom, num_x_tiles, num_y_tiles),
                Tilemap.from_2d(vy, self._cfg.target_zoom, num_x_tiles, num_y_tiles))

    @staticmethod
    def _get_tilemap_file_name(cfg: Config):
        return (f"{cfg.output_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.area=}-{cfg.sample_rate=}-{cfg.base_zoom=}.pkl")

    def get_force(self, p: Params) -> Vec2:
        top_left_tile = mercantile.tile(self._cfg.area.top_right.lon,
                                        self._cfg.area.bot_left.lat, zoom=self._cfg.target_zoom)

        active_tile = mercantile.tile(p.lon, p.lat, zoom=self._cfg.target_zoom)

        idx_x = active_tile.x - top_left_tile.x
        idx_y = active_tile.y - top_left_tile.y

        return self._vector_map[idx_y][idx_x]
