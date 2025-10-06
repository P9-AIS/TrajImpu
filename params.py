from dataclasses import dataclass


@dataclass
class Params:
    time: int = 0
    lat: float = 0
    lon: float = 0
    sog: float = 0
    cog: float = 0
    heading: float = 0
