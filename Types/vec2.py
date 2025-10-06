from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)


def create_vec2_array(vx: np.ndarray, vy: np.ndarray) -> list[list[Vec2]]:
    if vx.shape != vy.shape:
        raise ValueError("Input arrays must have the same shape.")

    rows, cols = vx.shape
    vec2_array = []

    for r in range(rows):
        row_list = []
        for c in range(cols):
            vec = Vec2(x=vx[r, c], y=vy[r, c])
            row_list.append(vec)
        vec2_array.append(row_list)

    return vec2_array
