from dataclasses import dataclass
from ForceTypes.vessel_types import VesselType


@dataclass
class AISStats:
    seq_len: int
    num_trajs: int
    num_records: int
    mean_lat: float
    mean_lon: float
    std_lat: float
    std_lon: float

    def combine(self, other: "AISStats") -> "AISStats":
        combined_num_trajs = self.num_trajs + other.num_trajs
        combined_num_records = self.num_records + other.num_records

        combined_mean_lat = (
            (self.mean_lat * self.num_records + other.mean_lat * other.num_records)
            / combined_num_records
        )
        combined_mean_lon = (
            (self.mean_lon * self.num_records + other.mean_lon * other.num_records)
            / combined_num_records
        )

        combined_var_lat = (
            self.num_records * (self.std_lat**2 + (self.mean_lat - combined_mean_lat)**2)
            + other.num_records * (other.std_lat**2 + (other.mean_lat - combined_mean_lat)**2)
        ) / combined_num_records

        combined_var_lon = (
            self.num_records * (self.std_lon**2 + (self.mean_lon - combined_mean_lon)**2)
            + other.num_records * (other.std_lon**2 + (other.mean_lon - combined_mean_lon)**2)
        ) / combined_num_records

        combined_std_lat = combined_var_lat**0.5
        combined_std_lon = combined_var_lon**0.5

        return AISStats(
            seq_len=self.seq_len,
            num_trajs=combined_num_trajs,
            num_records=combined_num_records,
            mean_lat=combined_mean_lat,
            mean_lon=combined_mean_lon,
            std_lat=combined_std_lat,
            std_lon=combined_std_lon,
        )
