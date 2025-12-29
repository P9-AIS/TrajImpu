from dataclasses import dataclass


@dataclass
class Params:
    time: int = 0
    northerns: float = 0
    easterns: float = 0
    sog: float = 0
    cog: float = 0
    rot: float = 0
    heading: float = 0
