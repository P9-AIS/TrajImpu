from Types.tilemap import Tilemap
from Types.vec2 import Vec2  # Assuming Vec2 has a magnitude() method
import os
from DataAccess.i_data_access_handler import AreaTuple
from Types.vec2 import Vec2  # Assuming Vec2 is defined elsewhere
import datetime
from ForceProviders.i_force_provider import IForceProvider, Params
from ForceProviders.traffic_force_provider import TrafficForceProvider
import mercantile
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from Types.vec2 import Vec2
from PIL import Image


@dataclass
class Config:
    vectormap: tuple[Tilemap[float], Tilemap[float]]
    output_dir: str = "Outputs/Heatmaps"
    target_pixel_size: int = 4000


# def generate_heatmap_image(cfg):
#     os.makedirs(cfg.output_dir, exist_ok=True)

#     num_x_tiles = cfg.vectormap[0].max_x_tile - cfg.vectormap[0].min_x_tile + 1
#     num_y_tiles = cfg.vectormap[0].max_y_tile - cfg.vectormap[0].min_y_tile + 1

#     force_magnitudes = np.zeros((num_y_tiles, num_x_tiles), dtype=np.float32)
#     for (x, y), u in cfg.vectormap[0].items():
#         v = cfg.vectormap[1][(x, y)]
#         magnitude = np.sqrt(u**2 + v**2)
#         force_magnitudes[y - cfg.vectormap[0].min_y_tile,
#                          x - cfg.vectormap[0].min_x_tile] = magnitude

#     normalized = 255 * (force_magnitudes - force_magnitudes.min()) / (np.ptp(force_magnitudes) + 1e-8)
#     normalized = normalized.astype(np.uint8)

#     from matplotlib.cm import viridis
#     colored = viridis(normalized / 255.0)
#     image_array = (colored[:, :, :3] * 255).astype(np.uint8)

#     img = Image.fromarray(image_array)

#     tile_size = max(cfg.target_pixel_size // max(num_x_tiles, num_y_tiles), 1)
#     if tile_size > 1:
#         img = img.resize((num_x_tiles * tile_size, num_y_tiles * tile_size), resample=Image.NEAREST)

#     filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-heatmap.png"
#     output_path = os.path.join(cfg.output_dir, filename)
#     img.save(output_path)

#     print(f"Heatmap saved to {output_path}")


def generate_heatmap_image(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)

    num_x_tiles = cfg.vectormap[0].max_x_tile - cfg.vectormap[0].min_x_tile + 1
    num_y_tiles = cfg.vectormap[0].max_y_tile - cfg.vectormap[0].min_y_tile + 1

    # Initialize empty map
    force_magnitudes = np.zeros((num_y_tiles, num_x_tiles), dtype=np.float32)

    # Fill magnitudes
    for (x, y), u in cfg.vectormap[0].items():
        v = cfg.vectormap[1][x, y]
        magnitude = np.sqrt(u**2 + v**2)
        force_magnitudes[y - cfg.vectormap[0].min_y_tile,
                         x - cfg.vectormap[0].min_x_tile] = magnitude

    low, high = np.percentile(force_magnitudes[force_magnitudes > 0], [1, 99])
    normalized = np.clip((force_magnitudes - low) / (high - low), 0, 1)

    cmap = plt.get_cmap('inferno')  # great dynamic range
    colored = cmap(normalized)
    colored[..., 3] = (normalized > 0).astype(float)  # transparency for zero values
    # img = Image.fromarray((colored * 255).astype(np.uint8), mode='RGBA')
    img = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    # Apply colormap (e.g. 'plasma', 'viridis', 'inferno', 'jet')
    # cmap = plt.get_cmap('plasma')
    # colored = cmap(normalized)  # Returns RGBA values in range [0, 1]

    # # Convert to 8-bit RGB image
    # img = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    # Optional scaling
    tile_size = max(cfg.target_pixel_size // max(num_x_tiles, num_y_tiles), 1)
    if tile_size > 1:
        img = img.resize(
            (num_x_tiles * tile_size, num_y_tiles * tile_size),
            resample=Image.NEAREST
        )

    # Save image
    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-heatmap.png"
    output_path = os.path.join(cfg.output_dir, filename)
    img.save(output_path)

    print(f"✅ Heatmap saved to {output_path}")


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
