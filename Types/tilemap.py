from collections import defaultdict

from typing import Generic, TypeVar, Tuple
import math
from Types.area import Area
from Utils.geo_converter import GeoConverter as gc


T = TypeVar("Numeric", int, float)


class Tilemap(Generic[T]):
    _tilemap: dict[int, T]
    _tile_size: int
    _espg3034_bounds: Area
    _dim_x: int
    _dim_y: int
    _E0: float
    _N0: float

    def __init__(self, tile_size: int, espg3034_bounds: Area):
        self._tile_size = tile_size
        self._tilemap = defaultdict(int)

        self._espg3034_bounds = espg3034_bounds

        offset_x, offset_y = gc.epsg3034_to_cell(espg3034_bounds.bottom_left.E,
                                                 espg3034_bounds.bottom_left.N, 0, 0, tile_size)
        self._E0, self._N0 = gc.cell_to_epsg3034(offset_x + 1, offset_y + 1, 0, 0, tile_size)

        max_x, max_y = gc.epsg3034_to_cell(espg3034_bounds.top_right.E,
                                           espg3034_bounds.top_right.N, self._E0, self._N0, tile_size)
        self._dim_x, self._dim_y = max_x + 1, max_y + 1

    def __getitem__(self, key: tuple[int, int]) -> T:
        return self._tilemap[Tilemap._tile_to_key(*key)]

    def __setitem__(self, key: tuple[int, int], value: T):
        self._tilemap[Tilemap._tile_to_key(*key)] = value

    def __len__(self) -> int:
        return len(self._tilemap)

    def get_tile_size(self) -> int:
        return self._tile_size

    def get_dimensions(self) -> Tuple[int, int]:
        return self._dim_x, self._dim_y

    def items(self):
        for key, val in self._tilemap.items():
            x, y = Tilemap._key_to_tile(key)
            yield (x, y), val

    def increment(self, x, y, n=1):
        key = Tilemap._tile_to_key(x, y)
        self._tilemap[key] = self._tilemap.get(key, 0) + n

    def tile_from_espg4326(self, lon, lat) -> Tuple[int, int]:
        x, y = gc.epsg3034_to_cell(*gc.espg4326_to_epsg3034(lon, lat), self._E0, self._N0, self._tile_size)
        return x, y

    def increment_espg4326(self, lon, lat, n=1):
        x, y = self.tile_from_espg4326(lon, lat)
        self.increment(x, y, n)

    def downscale_tile_map(self, factor: int) -> "Tilemap[T]":
        if factor < 1:
            raise ValueError(f"Downscale factor must be >= 1, got {factor}")

        if factor == 1:
            return self

        if math.sqrt(factor) % 1 != 0:
            raise ValueError(f"Downscale factor must be a perfect square, got {factor}")

        tile_scale_factor = int(math.sqrt(factor))

        if self._dim_x % tile_scale_factor != 0 or self._dim_y % tile_scale_factor != 0:
            print(
                f"Warning: Tilemap dimensions ({self._dim_x}, {self._dim_y}) are not divisible by factor {tile_scale_factor}")
            print(f"Resulting tilemap will be cropped to ({self._dim_x // tile_scale_factor}, "
                  f"{self._dim_y // tile_scale_factor})")

        new_tilemap = Tilemap(tile_size=self._tile_size * tile_scale_factor, espg3034_bounds=self._espg3034_bounds)
        new_dim_x, new_dim_y = new_tilemap.get_dimensions()

        for (x, y), val in self.items():
            new_x = x // tile_scale_factor
            new_y = y // tile_scale_factor

            if new_x >= new_dim_x or new_y >= new_dim_y:
                continue

            new_tilemap[new_x, new_y] += val

        return new_tilemap

    @staticmethod
    def from_2d(tile_array: list[list[T]], tile_size: int, espg3034_bounds: Area) -> "Tilemap[T]":
        tilemap = Tilemap(tile_size, espg3034_bounds)
        for y, row in enumerate(tile_array):
            for x, val in enumerate(row):
                if val != 0:
                    tilemap[x, y] = val
        return tilemap

    @staticmethod
    def _tile_to_key(x: int, y: int) -> int:
        key = (x << 32) | y
        return key

    @staticmethod
    def _key_to_tile(key: int) -> Tuple[int, int]:
        x = (key >> 32) & 0xFFFFFFFF
        y = key & 0xFFFFFFFF
        return x, y
