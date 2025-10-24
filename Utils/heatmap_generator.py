from Types.tilemap import Tilemap
from Types.vec2 import Vec3
import os
from Types.vec2 import Vec3
import datetime
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from Types.vec2 import Vec3
from PIL import Image


@dataclass
class Config:
    output_dir: str = "Outputs/Heatmaps"
    target_pixel_size: int = 4000
    color_map: str = 'jet'


def generate_heatmap_image(vectormap: tuple[np.ndarray, np.ndarray], cfg):
    print("Generating heatmap image...")

    os.makedirs(cfg.output_dir, exist_ok=True)

    vx, vy = vectormap
    force_magnitudes = np.sqrt(vx**2 + vy**2)
    force_magnitudes = np.flipud(force_magnitudes)
    force_magnitudes /= force_magnitudes.max()

    cmap = plt.get_cmap(cfg.color_map)
    colored = cmap(force_magnitudes)
    colored[..., 3] = (force_magnitudes > 0).astype(float)

    img = Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8), mode='RGB')

    num_y_tiles, num_x_tiles = vx.shape
    tile_size = max(cfg.target_pixel_size // max(num_x_tiles, num_y_tiles), 1)
    if tile_size > 1:
        img = img.resize(
            (num_x_tiles * tile_size, num_y_tiles * tile_size),
            resample=Image.NEAREST
        )

    filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-heatmap.png"
    output_path = os.path.join(cfg.output_dir, filename)
    img.save(output_path)

    print(f"✅ Heatmap saved to {output_path}")


def generate_vectormap(ff: list[list[Vec3]], tile_size: int = 25, output_dir: str = "Outputs/Vectormaps"):
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
