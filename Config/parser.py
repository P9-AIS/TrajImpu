import yaml
from Config.visitor import ConfigVisitorRegistry
from DataAccess.postgres_connection import Config as PostgresConfig
from ForceProviders.force_provider_traffic import Config as TrafficForceProviderConfig
from ForceProviders.force_provider_depth import Config as DepthForceProviderConfig
from Types.espg3034_coord import Espg3034Coord
from Types.area import Area
from Utils.heatmap_generator import Config as HeatmapGeneratorConfig
from dataclasses import dataclass


@dataclass
class Config:
    postgresCfg: PostgresConfig
    trafficForceProviderCfg: TrafficForceProviderConfig
    depthForceProviderCfg: DepthForceProviderConfig
    heatmapGeneratorCfg: HeatmapGeneratorConfig


def parse_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return Config(
        postgresCfg=ConfigVisitorRegistry.visit(PostgresConfig, cfg_dict["postgresCfg"]),
        trafficForceProviderCfg=ConfigVisitorRegistry.visit(
            TrafficForceProviderConfig, cfg_dict["trafficForceProviderCfg"]),
        depthForceProviderCfg=ConfigVisitorRegistry.visit(
            DepthForceProviderConfig, cfg_dict["depthForceProviderCfg"]),
        heatmapGeneratorCfg=ConfigVisitorRegistry.visit(HeatmapGeneratorConfig, cfg_dict["heatmapGeneratorCfg"]),
    )


ConfigVisitorRegistry.register(
    PostgresConfig,
    lambda data: PostgresConfig(**data)
)

ConfigVisitorRegistry.register(
    TrafficForceProviderConfig,
    lambda data: TrafficForceProviderConfig(
        start_date=data["start_date"],
        end_date=data["end_date"],
        sample_rate=data["sample_rate"],
        area=ConfigVisitorRegistry.visit(Area, data["area"]),
        vessel_types=data["vessel_types"],
        base_tile_size_m=data["base_tile_size_m"],
        down_scale_factor=data["down_scale_factor"],
        output_dir=data["output_dir"],
        sato_sigmas=data["sato_sigmas"],
        gaussian_sigma=data["gaussian_sigma"],
        low_percentile_cutoff=data["low_percentile_cutoff"],
        high_percentile_cutoff=data["high_percentile_cutoff"],
        sensitivity1=data["sensitivity1"],
        sensitivity2=data["sensitivity2"],
    )
)

ConfigVisitorRegistry.register(
    DepthForceProviderConfig,
    lambda data: DepthForceProviderConfig(
        area=ConfigVisitorRegistry.visit(Area, data["area"]),
        down_scale_factor=data["down_scale_factor"],
        output_dir=data["output_dir"],
        gaussian_sigma=data["gaussian_sigma"],
        low_percentile_cutoff=data["low_percentile_cutoff"],
        high_percentile_cutoff=data["high_percentile_cutoff"],
        sensitivity1=data["sensitivity1"],
        sensitivity2=data["sensitivity2"],
    )
)

ConfigVisitorRegistry.register(
    HeatmapGeneratorConfig,
    lambda data: HeatmapGeneratorConfig(**data)
)

ConfigVisitorRegistry.register(
    Area,
    lambda data: Area(ConfigVisitorRegistry.visit(Espg3034Coord, data["bottom_left"]),
                      ConfigVisitorRegistry.visit(Espg3034Coord, data["top_right"]))
)

ConfigVisitorRegistry.register(
    Espg3034Coord,
    lambda data: Espg3034Coord(**data)
)
