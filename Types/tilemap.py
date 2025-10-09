from collections import defaultdict

import mercantile
from typing import Generic, TypeVar, Tuple

T = TypeVar("Numeric", int, float)


class Tilemap(Generic[T]):
    tilemap: dict[int, T]
    base_zoom: int
    min_x_tile: int
    max_x_tile: int
    min_y_tile: int
    max_y_tile: int

    def __init__(self, base_zoom: int, min_x_tile: int, max_x_tile: int, min_y_tile: int, max_y_tile: int):
        self.base_zoom = base_zoom
        self.tilemap = defaultdict(int)
        self.min_x_tile = min_x_tile
        self.max_x_tile = max_x_tile
        self.min_y_tile = min_y_tile
        self.max_y_tile = max_y_tile

    def __getitem__(self, key: tuple[int, int]) -> T:
        x, y = key
        packed = (x << 32) | y
        return self.tilemap[packed]

    def __setitem__(self, key: tuple[int, int], value: T):
        x, y = key
        packed = (x << 32) | y
        self.tilemap[packed] = value

    def __len__(self) -> int:
        return len(self.tilemap)

    def items(self):
        for packed_key, val in self.tilemap.items():
            x = (packed_key >> 32) & 0xFFFFFFFF
            y = packed_key & 0xFFFFFFFF
            yield (x, y), val

    def increment(self, x, y, n=1):
        key = (x << 32) | y
        self.tilemap[key] = self.tilemap.get(key, 0) + n

    def downscale_tile_map(self, target_zoom: int):
        if target_zoom > self.base_zoom:
            raise ValueError(f"Target zoom ({target_zoom}) cannot be greater than base zoom ({self.base_zoom})")

        bot_left_tile = mercantile.Tile(x=self.min_x_tile, y=self.max_y_tile, z=self.base_zoom)
        top_right_tile = mercantile.Tile(x=self.max_x_tile, y=self.min_y_tile, z=self.base_zoom)

        bot_left_coord = mercantile.ul(bot_left_tile)
        top_right_coord = mercantile.ul(top_right_tile)

        new_bot_left_tile = mercantile.tile(bot_left_coord.lng, bot_left_coord.lat, zoom=target_zoom)
        new_top_right_tile = mercantile.tile(top_right_coord.lng, top_right_coord.lat, zoom=target_zoom)

        parent_counts = Tilemap(base_zoom=target_zoom, min_x_tile=new_bot_left_tile.x, max_x_tile=new_top_right_tile.x,
                                min_y_tile=new_top_right_tile.y, max_y_tile=new_bot_left_tile.y)

        for (x, y), val in self.items():
            tile = mercantile.Tile(x=x, y=y, z=self.base_zoom)
            parent = mercantile.parent(tile, zoom=target_zoom)
            parent_counts[parent.x, parent.y] += val

        return parent_counts

    @staticmethod
    def from_2d(tile_array: list[list[T]], base_zoom: int, x_offset: int, y_offset: int) -> "Tilemap[T]":
        tilemap = Tilemap(base_zoom, x_offset, x_offset + len(tile_array[0]), y_offset, y_offset + len(tile_array))
        for y, row in enumerate(tile_array):
            for x, val in enumerate(row):
                if val != 0:
                    tilemap[x + x_offset, y + y_offset] = val
        return tilemap
