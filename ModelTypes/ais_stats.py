from dataclasses import dataclass


@dataclass
class AISStats:
    seq_len: int
    num_attributes: int
    num_trajs: int
    num_records: int
    vessel_types: set[int]
    min_draught: float
    max_draught: float
    min_sog: float
    max_sog: float
    min_rot: float
    max_rot: float

    def combine(self, other: "AISStats") -> "AISStats":
        combined_num_trajs = self.num_trajs + other.num_trajs
        combined_num_records = self.num_records + other.num_records
        combined_vessel_types = self.vessel_types.union(other.vessel_types)
        combined_min_draught = min(self.min_draught, other.min_draught)
        combined_max_draught = max(self.max_draught, other.max_draught)
        combined_min_sog = min(self.min_sog, other.min_sog)
        combined_max_sog = max(self.max_sog, other.max_sog)
        combined_min_rot = min(self.min_rot, other.min_rot)
        combined_max_rot = max(self.max_rot, other.max_rot)

        return AISStats(
            seq_len=self.seq_len,
            num_attributes=self.num_attributes,
            num_trajs=combined_num_trajs,
            num_records=combined_num_records,
            vessel_types=combined_vessel_types,
            min_draught=combined_min_draught,
            max_draught=combined_max_draught,
            min_sog=combined_min_sog,
            max_sog=combined_max_sog,
            min_rot=combined_min_rot,
            max_rot=combined_max_rot
        )
