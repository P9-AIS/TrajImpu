import os
from vec2 import Vec2  # Assuming Vec2 is defined elsewhere
import datetime
from ForceProviders.i_force_provider import IForceProvider, Params
from ForceProviders.traffic_force_provider import TrafficForceProvider
import mercantile
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from vec2 import Vec2


@dataclass
class Config:
    area_top_left_lat: float
    area_top_left_lon: float
    area_bottom_right_lat: float
    area_bottom_right_lon: float
    zoom_level: int


def traffic_force_field(fp: TrafficForceProvider, cfg: Config):
    top_left_tile = mercantile.tile(cfg.area_top_left_lon, cfg.area_top_left_lat, zoom=cfg.zoom_level)
    bottom_right_tile = mercantile.tile(cfg.area_bottom_right_lon, cfg.area_bottom_right_lat, zoom=cfg.zoom_level)

    num_x_tiles = bottom_right_tile.x - top_left_tile.x
    num_y_tiles = bottom_right_tile.y - top_left_tile.y

    force_map = []
    for y in range(num_y_tiles):
        row_counts = []
        for x in range(num_x_tiles):
            tile = mercantile.Tile(top_left_tile.x + x, top_left_tile.y + y, cfg.zoom_level)
            lng_lat = mercantile.ul(tile)
            heat = fp.get_force(Params(0, lng_lat.lat, lng_lat.lng, 0, 0, 0))
            row_counts.append(heat)
        force_map.append(row_counts)

    return force_map


def generate_heatmap(ff: list[list[Vec2]], tile_size: int = 25, output_dir: str = "Outputs/Heatmaps"):
    if not ff or not ff[0]:
        print("Input force field is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    force_magnitudes = np.array([[v.magnitude() for v in row] for row in ff])

    num_y_tiles, num_x_tiles = force_magnitudes.shape
    fig_width = num_x_tiles * tile_size / 100
    fig_height = num_y_tiles * tile_size / 100

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    norm = Normalize(vmin=force_magnitudes.min(), vmax=force_magnitudes.max())
    cbar_label = "Force Magnitude"

    heatmap = ax.imshow(force_magnitudes, cmap='viridis', norm=norm, origin='lower')

    cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')
    cbar.set_label(cbar_label)

    ax.set_title('Heatmap')
    ax.set_xlabel('Tile X-Coordinate')
    ax.set_ylabel('Tile Y-Coordinate')

    ax.set_xticks(np.arange(num_x_tiles))
    ax.set_yticks(np.arange(num_y_tiles))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-heatmap.png"

    plt.savefig(f"{output_dir}/{filename}", dpi=100)
    print(f"Heatmap saved to {output_dir}/{filename}")

    plt.close(fig)


def generate_vectormap(ff: list[list[Vec2]], tile_size: int = 25, output_dir: str = "Outputs/Vectormaps"):
    if not ff or not ff[0]:
        print("Input force field is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    u = np.array([[v.x for v in row] for row in ff])
    v = np.array([[v.y for v in row] for row in ff])

    num_y_tiles, num_x_tiles = u.shape

    x_coords, y_coords = np.meshgrid(np.arange(num_x_tiles), np.arange(num_y_tiles))

    fig_width = num_x_tiles * tile_size / 100
    fig_height = num_y_tiles * tile_size / 100

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.quiver(x_coords, y_coords, u, v, color='blue', angles='xy', scale_units='xy', scale=1)

    ax.set_title('Traffic Vector Field Map')
    ax.set_xlabel('Tile X-Coordinate')
    ax.set_ylabel('Tile Y-Coordinate')

    ax.set_xlim(-0.5, num_x_tiles - 0.5)
    ax.set_ylim(-0.5, num_y_tiles - 0.5)

    ax.set_xticks(np.arange(num_x_tiles))
    ax.set_yticks(np.arange(num_y_tiles))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-vectormap.png"
    plt.savefig(f"{output_dir}/{filename}", dpi=100)
    print(f"Vectormap saved to {output_dir}/{filename}")

    plt.close(fig)
