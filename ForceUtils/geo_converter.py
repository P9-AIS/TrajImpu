from datetime import date
import numpy as np
from pyproj import Transformer, Geod
import math
import torch
from ForceTypes.vec3 import Vec3


class GeoConverter:
    _to_espg4326 = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
    _from_espg4326 = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)

    @staticmethod
    def epsg3034_to_espg4326(E, N):
        """
        Convert projected coordinates (E,N) in EPSG:3034 to (lon, lat) in EPSG:4326.

        Parameters
        ----------
        E, N : float
            Easting and northing in metres (EPSG:3034)

        Returns
        -------
        (lon, lat) : tuple[float, float]
            Longitude and latitude in degrees (EPSG:4326)
        """
        lon, lat = GeoConverter._to_espg4326.transform(E, N)
        return lon, lat

    @staticmethod
    def espg4326_to_epsg3034(lon, lat):
        """
        Convert geographic coordinates (lon, lat) in EPSG:4326 to (E,N) in EPSG:3034.

        Parameters
        ----------
        lon, lat : float
            Longitude and latitude in degrees (EPSG:4326)

        Returns
        -------
        (E, N) : tuple[float, float]
            Projected coordinates in metres (EPSG:3034)
        """
        E, N = GeoConverter._from_espg4326.transform(lon, lat)
        return E, N

    @staticmethod
    def espg4326_to_epsg3034_batch(lon: np.ndarray, lat: np.ndarray):
        transformed = GeoConverter._from_espg4326.transform(lon, lat)
        ES = transformed[0].reshape(lon.shape)
        NS = transformed[1].reshape(lat.shape)
        return ES, NS

    @staticmethod
    def espg3034_to_epsg4326_batch(E: np.ndarray, N: np.ndarray):
        transformed = GeoConverter._to_espg4326.transform(E, N)
        lon = transformed[0].reshape(E.shape)
        lat = transformed[1].reshape(N.shape)
        return lon, lat

    @staticmethod
    def espg4326_to_epsg3034_batch_tensor(lon: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lon_cpu = lon.cpu().numpy()
        lat_cpu = lat.cpu().numpy()
        catted = np.column_stack((lon_cpu.flatten(), lat_cpu.flatten()))
        transformed = GeoConverter._from_espg4326.transform(catted[:, 0], catted[:, 1])
        ES = transformed[0].reshape(lon_cpu.shape)
        NS = transformed[1].reshape(lat_cpu.shape)
        return torch.tensor(ES, dtype=torch.double).to(lon.device), torch.tensor(NS, dtype=torch.double).to(lat.device)

    @staticmethod
    def epsg3034_to_espg4326_batch_tensor(E: torch.Tensor, N: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        E_cpu = E.cpu().numpy()
        N_cpu = N.cpu().numpy()
        catted = np.column_stack((E_cpu.flatten(), N_cpu.flatten()))
        transformed = GeoConverter._to_espg4326.transform(catted[:, 0], catted[:, 1])
        lon = transformed[0].reshape(E_cpu.shape)
        lat = transformed[1].reshape(N_cpu.shape)
        return torch.tensor(lon, dtype=torch.float).to(E.device), torch.tensor(lat, dtype=torch.float).to(N.device)

    @staticmethod
    def epsg3034_to_cell(E, N, E0, N0, cell_size=50.0, origin_is_cell_center=False):
        """
        Convert EPSG:3034 coordinates (E, N) to DDM 50x50m grid cell indices (x, y).

        Parameters
        ----------
        E, N : float
            Projected coordinates in EPSG:3034 (metres)
        E0, N0 : float
            Grid origin in EPSG:3034 (metres)
        cell_size : float
            Cell size in metres (default 50)
        origin_is_cell_center : bool
            If True, E0,N0 refer to the *centre* of cell (0,0).
            If False, E0,N0 refer to the *lower-left corner* of cell (0,0).

        Returns
        -------
        (x, y) : tuple[int, int]
            Integer cell indices.
        """
        if origin_is_cell_center:
            # Shift origin from center to corner
            E0_corner = E0 - 0.5 * cell_size
            N0_corner = N0 - 0.5 * cell_size
        else:
            E0_corner = E0
            N0_corner = N0

        x = int(math.floor((E - E0_corner) / cell_size))
        y = int(math.floor((N - N0_corner) / cell_size))

        return x, y

    @staticmethod
    def cell_to_epsg3034(x, y, E0, N0, cell_size=50.0, origin_is_cell_center=False):
        """
        Convert DDM 50x50m grid indices (x, y) to EPSG:3034 coordinates (E, N).

        x, y : integers (cell indices)
        E0, N0 : grid origin (EPSG:3034, metres)
        cell_size : grid cell size (default 50)
        origin_is_cell_center : whether E0,N0 are the center of cell (0,0)
        Returns (E, N) in metres (EPSG:3034)
        """
        if origin_is_cell_center:
            E = E0 + x * cell_size
            N = N0 + y * cell_size
        else:
            E = E0 + (x + 0.5) * cell_size
            N = N0 + (y + 0.5) * cell_size
        return E, N

    @staticmethod
    def cell_corners(x, y, E0, N0, cell_size=50.0, origin_is_cell_center=False):
        """
        Get the four corners of a grid cell (x, y) in EPSG:3034 coordinates (E, N).

        Parameters
        ----------
        x, y : int
            Cell indices.
        E0, N0 : float
            Grid origin in EPSG:3034 (metres)
        cell_size : float
            Cell size in metres (default 50)
        origin_is_cell_center : bool
            If True, E0,N0 refer to the *centre* of cell (0,0).
            If False, E0,N0 refer to the *lower-left corner* of cell (0,0).

        Returns
        -------
        corners : list[tuple[float, float]]
            List of four corners [(E1,N1), (E2,N2), (E3,N3), (E4,N4)] in metres (EPSG:3034)
            in the order: bottom-left, bottom-right, top-right, top-left.
        """
        if origin_is_cell_center:
            E0_corner = E0 - 0.5 * cell_size
            N0_corner = N0 - 0.5 * cell_size
        else:
            E0_corner = E0
            N0_corner = N0

        E1 = E0_corner + x * cell_size
        N1 = N0_corner + y * cell_size

        E2 = E1 + cell_size
        N2 = N1

        E3 = E2
        N3 = N2 + cell_size

        E4 = E1
        N4 = N3

        return [(E1, N1), (E2, N2), (E3, N3), (E4, N4)]
