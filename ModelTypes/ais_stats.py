from dataclasses import dataclass
from ForceTypes.vessel_types import VesselType


@dataclass
class AISStats:
    seq_len: int
    num_trajs: int
    num_records: int
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def combine(self, other: "AISStats") -> "AISStats":
        combined_num_trajs = self.num_trajs + other.num_trajs
        combined_num_records = self.num_records + other.num_records
        combined_min_lat = min(self.min_lat, other.min_lat)
        combined_max_lat = max(self.max_lat, other.max_lat)
        combined_min_lon = min(self.min_lon, other.min_lon)
        combined_max_lon = max(self.max_lon, other.max_lon)

        return AISStats(
            seq_len=self.seq_len,
            num_trajs=combined_num_trajs,
            num_records=combined_num_records,
            min_lat=combined_min_lat,
            max_lat=combined_max_lat,
            min_lon=combined_min_lon,
            max_lon=combined_max_lon,
        )
