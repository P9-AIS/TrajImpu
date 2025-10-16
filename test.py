# Requires: pip install pyproj
from pyproj import Transformer
import math


def epsg3034_to_wgs84(E, N):
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
    _to_wgs84 = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
    lon, lat = _to_wgs84.transform(E, N)
    return lon, lat


def wgs84_to_epsg3034(lon, lat):
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
    _from_wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    E, N = _from_wgs84.transform(lon, lat)
    return E, N


def epsg3034_to_cell(E, N, E0, N0, cell_size=50.0, origin_is_cell_center=True):
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


def cell_to_epsg3034(x, y, E0, N0, cell_size=50.0, origin_is_cell_center=True):
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


def cell_corners(x, y, E0, N0, cell_size=50.0, origin_is_cell_center=True):
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


# Example usage (you must replace E0,N0 with the real origin for your dataset):
if __name__ == "__main__":
    wgs84_bound_bottom_left = (-16.1, 33.26)
    wgs84_bound_top_right = (38.01, 84.73)

    offset_x, offset_y = epsg3034_to_cell(
        *wgs84_to_epsg3034(*wgs84_bound_bottom_left), 0, 0, origin_is_cell_center=True)
    E0, N0 = cell_to_epsg3034(offset_x, offset_y, 0, 0, origin_is_cell_center=True)
    max_x, max_y = epsg3034_to_cell(*wgs84_to_epsg3034(*wgs84_bound_top_right), E0, N0, origin_is_cell_center=True)
    dim_x, dim_y = max_x + 1, max_y + 1

    print(f"Grid origin (E0,N0) = ({E0}, {N0})")
    print(f"Grid dimensions = ({dim_x} x {dim_y}) cells")

    # lon, lat = 10.028850747018821, 58.26100602964127

    # lon, lat = 12.41151237487793, 56.17210388183594
    # lon, lat = 12.456230163574219, 56.14387130737305
    # lon, lat = 13.9693021774292, 54.981685638427734
    # lon = 3.541543017598474
    # lat = 54.21002006394076
    lon = 16.964639913892025
    lat = 58.087114961437905

    # E0 = 1599600.0   # <-- REPLACE with grid origin easting (EPSG:3034)
    # N0 = 1190450.0  # <-- REPLACE with grid origin northing (EPSG:3034)

    # E0 = 3592900.0
    # N0 = 3055500.0

    E0 = 0
    N0 = 0

    print("hey", cell_to_epsg3034(*epsg3034_to_cell(*wgs84_to_epsg3034(lon, lat), E0, N0), E0, N0))

    X_o = 71858 - 31992
    Y_o = 69507 - 23809

    E, N = wgs84_to_epsg3034(lon, lat)
    X_s, Y_s = epsg3034_to_cell(E, N, E0, N0, origin_is_cell_center=True)

    print(X_s, Y_s)

    X_l = X_s - X_o
    Y_l = Y_o - Y_s

    print(X_l, Y_l)

    dk_bl_x = X_o + 16132
    dk_bl_y = Y_o

    print(epsg3034_to_wgs84(*cell_to_epsg3034(dk_bl_x, dk_bl_y, E0, N0, origin_is_cell_center=True)))

# (31992, 23809)
