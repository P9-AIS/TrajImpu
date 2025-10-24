from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> "Vec3":
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float) -> "Vec3":
        return Vec3(self.x / other, self.y / other, self.z / other)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y, self.z)

    def normalize(self) -> "Vec3":
        mag = self.magnitude()
        if mag == 0:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)
