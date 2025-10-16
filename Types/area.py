from dataclasses import dataclass

from Types.espg3034_coord import Espg3034Coord


@dataclass
class Area:
    bottom_left: Espg3034Coord
    top_right: Espg3034Coord
