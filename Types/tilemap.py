import pickle
from typing import Generic, TypeVar, Tuple, Callable
import math

import numpy as np
from Types.area import Area
from Utils.geo_converter import GeoConverter as gc


T = TypeVar("Numeric", int, float)


class Tilemap(Generic[T]):
    _tilemap: np.ndarray
    _tile_size: int
    _espg3034_bounds: Area
    _dim_x: int
    _dim_y: int
    _E0: float
    _N0: float

    def __init__(self, tile_size: int, espg3034_bounds: Area):
        self._tile_size = tile_size
        self._espg3034_bounds = espg3034_bounds

        offset_x, offset_y = gc.epsg3034_to_cell(espg3034_bounds.bottom_left.E,
                                                 espg3034_bounds.bottom_left.N, 25, 25, tile_size)
        self._E0, self._N0 = gc.cell_to_epsg3034(offset_x, offset_y, 25, 25, tile_size)

        max_x, max_y = gc.epsg3034_to_cell(espg3034_bounds.top_right.E,
                                           espg3034_bounds.top_right.N, self._E0, self._N0, tile_size)
        self._dim_x, self._dim_y = max_x + 1, max_y + 1
        self._tilemap = np.zeros((self._dim_y, self._dim_x), dtype=np.int32)

    def __getitem__(self, key: tuple[int, int]) -> T:
        return self._tilemap[*key]

    def __setitem__(self, key: tuple[int, int], value: T):
        self._tilemap[*key] = value

    def get_tile_size(self) -> int:
        return self._tile_size

    def get_dimensions(self) -> Tuple[int, int]:
        return self._dim_x, self._dim_y

    def get_espg3034_bounds(self) -> Area:
        return self._espg3034_bounds

    def get_array(self) -> np.ndarray:
        return self._tilemap

    def update_tile(self, x, y, func: Callable[[T], T]):
        self._tilemap[y, x] = func(self._tilemap[y, x])

    def set_tile_from_espg3034(self, E, N, value: T):
        x, y = self.tile_from_espg3034(E, N)
        self[y, x] = value

    def set_tile_from_espg4326(self, lon, lat, value: T):
        x, y = self.tile_from_espg4326(lon, lat)
        self[y, x] = value

    def tile_from_espg3034(self, E, N) -> Tuple[int, int]:
        x, y = gc.epsg3034_to_cell(E, N, self._E0, self._N0, self._tile_size)
        return x, y

    def tile_from_espg4326(self, lon, lat) -> Tuple[int, int]:
        x, y = gc.epsg3034_to_cell(*gc.espg4326_to_epsg3034(lon, lat), self._E0, self._N0, self._tile_size)
        return x, y

    def update_tile_espg4326(self, lon, lat, func: Callable[[T], T]):
        x, y = self.tile_from_espg4326(lon, lat)
        self.update_tile(x, y, func)

    def update_tile_espg3034(self, E, N, func: Callable[[T], T]):
        x, y = self.tile_from_espg3034(E, N)
        self.update_tile(x, y, func)

    def downscale(self, factor: int, aggregation_func: Callable[[np.ndarray], np.ndarray] = np.sum):
        if factor < 1:
            raise ValueError("Factor must be >= 1")

        scale = int(math.sqrt(factor))

        if scale * scale != factor:
            raise ValueError("Factor must be a perfect square")

        new_dim_y = self._tilemap.shape[0] // scale * scale
        new_dim_x = self._tilemap.shape[1] // scale * scale
        arr_cropped = self._tilemap[:new_dim_y, :new_dim_x]

        print(f"Downscaled tilemap to {new_dim_x // scale}x{new_dim_y // scale} using factor {factor}")

        reshaped = arr_cropped.reshape(new_dim_y // scale, scale, new_dim_x // scale, scale)
        downscaled = aggregation_func(reshaped, axis=(1, 3))

        self._tilemap = downscaled
        self._dim_y, self._dim_x = downscaled.shape
        self._tile_size *= scale

    def save(self, path: str):
        print(f"Saving tile map to '{path}'")
        np.savez_compressed(
            path,
            tilemap_object=pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL),
            tile_array=self._tilemap
        )
        print(f"Saved tile map of size {self.get_dimensions()}\n")

    @staticmethod
    def load(path: str) -> "Tilemap":
        print(f"Loading tile map from '{path}'")
        with np.load(path, allow_pickle=True) as data:
            tilemap: Tilemap = pickle.loads(data['tilemap_object'].item())
            tilemap._tilemap = data['tile_array']
        print(f"Loaded tile map of size {tilemap.get_dimensions()}\n")
        return tilemap
