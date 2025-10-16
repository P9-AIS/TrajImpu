from dataclasses import dataclass

from Types.lonlat import LonLat


@dataclass
class Area:
    bottom_left: LonLat
    top_right: LonLat
