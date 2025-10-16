import math
import matplotlib.pyplot as plt
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
from skimage.filters import sato
from scipy.ndimage import gaussian_filter


@dataclass
class Config:
    start_date: dt.date
    end_date: dt.date
    sample_rate: int
    area: Area
    vessel_types: list[VesselType]
    base_tile_size_m: int = 50
    down_scale_factor: int = 1
    output_dir: str = "Outputs/Tilemaps"
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

        tile_map = tile_map.downscale_tile_map(self._cfg.down_scale_factor)

        print(
            f"Downsampled tile map to {self._cfg.base_tile_size_m * math.sqrt(self._cfg.down_scale_factor)}m, now contains {len(tile_map)} tiles")

        self._vectormap = self._get_vector_map(tile_map)

    def _get_tile_map(self):
        days: list[dt.date] = []

        cur_date = self._cfg.start_date
        while cur_date < self._cfg.end_date:
            days.append(cur_date)
            cur_date = cur_date + dt.timedelta(self._cfg.sample_rate)

        ais_messages = self._data_handler.get_ais_messages_no_stops(days, self._cfg.area)

        print(f"Creating {self._cfg.base_tile_size_m}m tile map from {len(ais_messages)} AIS messages")

        tile_map = Tilemap(self._cfg.base_tile_size_m, self._cfg.area)

        for (lon, lat) in tqdm(ais_messages, desc="Aggregating tiles"):
            tile_map.increment_espg4326(lon, lat)

        return tile_map

    def _get_vector_map(self, tile_map):
        print(f"Creating vector field from tile map")

        Z = np.zeros(tile_map.get_dimensions(), dtype=np.float32)

        for (x, y), count in tqdm(tile_map.items(), total=len(tile_map), desc="Building vector field"):
            Z[x, y] = count

        TrafficForceProvider._save_distribution_plots(Z, "Outputs/Distributions", low_cut=self._cfg.low_percentile_cutoff, high_cut=self._cfg.high_percentile_cutoff,
                                                      sensitivity=self._cfg.sensitivity1, prefix="Z_distribution")

        low, high = np.percentile(Z[Z > 0], [self._cfg.low_percentile_cutoff, self._cfg.high_percentile_cutoff])
        Z_norm = np.clip((Z - low) / (high - low), 0, 1)
        Z_norm = Z_norm ** (1 / self._cfg.sensitivity1)

        # Apply vesselness filter (Sato) to enhance linear structures
        Z_vessel = sato(Z_norm, sigmas=self._cfg.sato_sigmas, black_ridges=False)
        Z_vessel /= Z_vessel.max()

        Z_smooth = gaussian_filter(Z_vessel, sigma=self._cfg.gaussian_sigma)
        Z_smooth /= Z_smooth.max()
        Z_smooth = Z_smooth ** (1 / self._cfg.sensitivity2)

        dz_dy, dz_dx = np.gradient(Z_smooth)
        grad_mag = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-8
        vx = -dz_dx / grad_mag * Z_smooth
        vy = -dz_dy / grad_mag * Z_smooth

        return (
            Tilemap.from_2d(vx, self._cfg.base_tile_size_m, self._cfg.area),
            Tilemap.from_2d(vy, self._cfg.base_tile_size_m, self._cfg.area)
        )

    @staticmethod
    def _get_tilemap_file_name(cfg: Config):
        return (f"{cfg.output_dir}/{cfg.start_date=}-{cfg.end_date=}-{cfg.sample_rate=}-{cfg.base_tile_size_m=}.pkl")

    def get_force(self, p: Params) -> Vec2:
        x, y = self._vector_map[0].tile_from_espg4326(p.lon, p.lat)

        x_force = self._vector_map[0][x, y]
        y_force = self._vector_map[1][x, y]

        return Vec2(x_force, y_force)

    @staticmethod
    def _save_distribution_plots(Z, output_dir, low_cut=2, high_cut=99.9, sensitivity=3, prefix="Z_distribution"):
        os.makedirs(output_dir, exist_ok=True)

        values = Z[Z > 0].flatten()
        low, high = np.percentile(values, [low_cut, high_cut])
        Z_norm = np.clip((Z - low) / (high - low), 0, 1)
        Z_norm = Z_norm ** (1 / sensitivity)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # 1️⃣ Histogram (log freq)
        axs[0].hist(values, bins=300, log=True, color='steelblue', alpha=0.8)
        axs[0].axvline(low, color='orange', linestyle='--', label=f"low {low_cut}%")
        axs[0].axvline(high, color='red', linestyle='--', label=f"high {high_cut}%")
        axs[0].set_title("Original value distribution (log freq)")
        axs[0].set_xlabel("Tile value")
        axs[0].set_ylabel("Frequency (log)")
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # 2️⃣ CDF (Cumulative Distribution)
        sorted_vals = np.sort(values)
        cdf = np.linspace(0, 100, len(sorted_vals))
        axs[1].plot(sorted_vals, cdf, color='steelblue')
        axs[1].axvline(low, color='orange', linestyle='--', label=f"low {low_cut}%")
        axs[1].axvline(high, color='red', linestyle='--', label=f"high {high_cut}%")
        axs[1].set_title("Cumulative Distribution (CDF)")
        axs[1].set_xlabel("Tile value")
        axs[1].set_ylabel("Percentile")
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        # 3️⃣ Histogram after normalization
        axs[2].hist(Z_norm[Z_norm > 0].flatten(), bins=300, color='green', alpha=0.8)
        axs[2].set_title(f"After normalization (sensitivity={sensitivity})")
        axs[2].set_xlabel("Normalized value (0–1)")
        axs[2].set_ylabel("Frequency")
        axs[2].grid(alpha=0.3)

        plt.tight_layout()

        # 4️⃣ Save
        out_path = os.path.join(output_dir, f"{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{prefix}.png")
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"📊 Distribution plot saved to: {out_path}")
