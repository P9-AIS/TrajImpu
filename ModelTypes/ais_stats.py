from dataclasses import dataclass


@dataclass
class AISStats:
    seq_len: int
    num_trajs: int
    num_records: int
    num_masked_values: int

    mean_abs_delta_n: float
    mean_abs_delta_e: float
    std_delta_n: float
    std_delta_e: float

    min_dist: float
    max_dist: float
    mean_dist: float
    std_dist: float

    min_traj_len: float
    max_traj_len: float
    mean_traj_len: float
    std_traj_len: float

    min_masked_len: float
    max_masked_len: float
    mean_masked_len: float
    std_masked_len: float

    min_duration: float
    max_duration: float
    mean_duration: float
    std_duration: float

    min_traj_duration: float
    max_traj_duration: float
    mean_traj_duration: float
    std_traj_duration: float

    min_masked_duration: float
    max_masked_duration: float
    mean_masked_duration: float
    std_masked_duration: float

    def __str__(self) -> str:
        lines = []
        add = lines.append

        add("=== AIS Statistics ===")
        add(f"{'Seq length:':20} {self.seq_len}")
        add(f"{'Num trajectories:':20} {self.num_trajs}")
        add(f"{'Num records:':20} {self.num_records}")
        add(f"{'Masked values:':20} {self.num_masked_values}")
        add("")

        add("— ΔN / ΔE —")
        add(f"{'Mean abs ΔN:':20} {self.mean_abs_delta_n:.4f}")
        add(f"{'Mean abs ΔE:':20} {self.mean_abs_delta_e:.4f}")
        add(f"{'Std ΔN:':20} {self.std_delta_n:.4f}")
        add(f"{'Std ΔE:':20} {self.std_delta_e:.4f}")
        add("")

        add("— Distances —")
        add(f"{'Min dist:':20} {self.min_dist:.4f}")
        add(f"{'Max dist:':20} {self.max_dist:.4f}")
        add(f"{'Mean dist:':20} {self.mean_dist:.4f}")
        add(f"{'Std dist:':20} {self.std_dist:.4f}")
        add("")

        add("— Trajectory lengths —")
        add(f"{'Min traj len:':20} {self.min_traj_len:.2f}")
        add(f"{'Max traj len:':20} {self.max_traj_len:.2f}")
        add(f"{'Mean traj len:':20} {self.mean_traj_len:.2f}")
        add(f"{'Std traj len:':20} {self.std_traj_len:.2f}")
        add("")

        add("— Masked segment lengths —")
        add(f"{'Min masked len:':20} {self.min_masked_len:.2f}")
        add(f"{'Max masked len:':20} {self.max_masked_len:.2f}")
        add(f"{'Mean masked len:':20} {self.mean_masked_len:.2f}")
        add(f"{'Std masked len:':20} {self.std_masked_len:.2f}")
        add("")

        add("— Durations —")
        add(f"{'Min duration:':20} {self.min_duration:.2f}")
        add(f"{'Max duration:':20} {self.max_duration:.2f}")
        add(f"{'Mean duration:':20} {self.mean_duration:.2f}")
        add(f"{'Std duration:':20} {self.std_duration:.2f}")
        add("")

        add("— Trajectory durations —")
        add(f"{'Min traj duration:':20} {self.min_traj_duration:.2f}")
        add(f"{'Max traj duration:':20} {self.max_traj_duration:.2f}")
        add(f"{'Mean traj duration:':20} {self.mean_traj_duration:.2f}")
        add(f"{'Std traj duration:':20} {self.std_traj_duration:.2f}")
        add("")

        add("— Masked segment durations —")
        add(f"{'Min masked duration:':20} {self.min_masked_duration:.2f}")
        add(f"{'Max masked duration:':20} {self.max_masked_duration:.2f}")
        add(f"{'Mean masked duration:':20} {self.mean_masked_duration:.2f}")
        add(f"{'Std masked duration:':20} {self.std_masked_duration:.2f}")

        return "\n".join(lines)
