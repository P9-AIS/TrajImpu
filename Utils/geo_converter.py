from datetime import date
from pyproj import Transformer, Geod
import math
import wmm2020

from Types.vec2 import Vec3


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

    @staticmethod
    def normalize_angle(angle_deg):
        """Normalize to [0, 360)."""
        return (angle_deg % 360.0 + 360.0) % 360.0

    @staticmethod
    def move_along_mag_heading(heading_mag_deg, distance_m, lon, lat, dt):
        """
        Move from a starting position (lat, lon) using a magnetic heading.
        Accounts for magnetic declination (WMM2020) and Earth's shape.
        Returns new (lat2, lon2).
        """
        # --- magnetic → true ---
        year_decimal = dt.year + (dt.timetuple().tm_yday / 365.25)
        result = wmm2020.wmm(lat, lon, 0, year_decimal)

        decl = float(result["decl"].values.item())
        heading_true = heading_mag_deg + decl

        # --- move along true heading ---
        geod = Geod(ellps="WGS84")
        lon2, lat2, _ = geod.fwd(lon, lat, heading_true, distance_m)
        return lon2, lat2, heading_true, decl

    @staticmethod
    def espg4326_heading_to_espg3034_heading(heading_deg, lon, lat, dt):
        """
        Convert a heading in EPSG:4326 (degrees clockwise from true north)
        to a heading in EPSG:3034 (degrees clockwise from grid north).
        """
        espg4326_p1 = (lon, lat)
        espg4326_p2 = GeoConverter.move_along_mag_heading(heading_deg, 10, lon, lat, dt)[:2]

        espg3034_p1 = GeoConverter.espg4326_to_epsg3034(*espg4326_p1)
        espg3034_p2 = GeoConverter.espg4326_to_epsg3034(*espg4326_p2)

        heading_vec = Vec3(espg3034_p2[0] - espg3034_p1[0], espg3034_p2[1] - espg3034_p1[1], 0.0)
        return heading_vec.normalize()
