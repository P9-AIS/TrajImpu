import yaml
from Config.visitor import ConfigVisitorRegistry
from Connection.postgres_connection import Config as PostgresConfig
from ForceProviders.force_provider_traffic import Config as TrafficForceProviderConfig
from ForceProviders.force_provider_depth import Config as DepthForceProviderConfig
from ForceTypes.espg3034_coord import Espg3034Coord
from ForceTypes.area import Area
from ForceUtils.heatmap_generator import Config as HeatmapGeneratorConfig
from ForceData.force_data_upload_handler_http import Config as ForceDataUploadHandlerConfig
from Model.model import Config as ModelConfig
from ModelUtils.data_loader import Config as ModelDataLoaderConfig
from ModelData.model_data_access_handler_csv import Config as ModelDataConfig
from ModelUtils.data_processor import MaskingStrategy, Config as ModelProcessorConfig
from ModelPipeline.trainer import Config as ModelTrainerConfig
from ModelUtils.loss_calculator import Config as ModelLossConfig
from ModelData.model_data_upload_handler_http import Config as ModelDataUploadHandlerConfig
from dataclasses import dataclass


@dataclass
class Config:
    postgresCfg: PostgresConfig
    trafficForceProviderCfg: TrafficForceProviderConfig
    depthForceProviderCfg: DepthForceProviderConfig
    heatmapGeneratorCfg: HeatmapGeneratorConfig
    forceDataUploadHandlerCfg: ForceDataUploadHandlerConfig
    modelCfg: ModelConfig
    modelDataLoaderCfg: ModelDataLoaderConfig
    modelDataCfg: ModelDataConfig
    modelDataProcessorCfg: ModelProcessorConfig
    modelTrainerCfg: ModelTrainerConfig
    modelLossCfg: ModelLossConfig
    modelDataUploadHandlerCfg: ModelDataUploadHandlerConfig


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
        forceDataUploadHandlerCfg=ConfigVisitorRegistry.visit(
            ForceDataUploadHandlerConfig, cfg_dict["forceDataUploadHandlerCfg"]),
        modelCfg=ConfigVisitorRegistry.visit(ModelConfig, cfg_dict["modelCfg"]),
        modelDataLoaderCfg=ConfigVisitorRegistry.visit(ModelDataLoaderConfig, cfg_dict["modelDataLoaderCfg"]),
        modelDataCfg=ConfigVisitorRegistry.visit(ModelDataConfig, cfg_dict["modelDataCfg"]),
        modelDataProcessorCfg=ConfigVisitorRegistry.visit(ModelProcessorConfig, cfg_dict["modelDataProcessorCfg"]),
        modelTrainerCfg=ConfigVisitorRegistry.visit(ModelTrainerConfig, cfg_dict["modelTrainerCfg"]),
        modelLossCfg=ConfigVisitorRegistry.visit(ModelLossConfig, cfg_dict["modelLossCfg"]),
        modelDataUploadHandlerCfg=ConfigVisitorRegistry.visit(
            ModelDataUploadHandlerConfig, cfg_dict["modelDataUploadHandlerCfg"])
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
        date_step=data["date_step"],
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

ConfigVisitorRegistry.register(
    ForceDataUploadHandlerConfig,
    lambda data: ForceDataUploadHandlerConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelDataLoaderConfig,
    lambda data: ModelDataLoaderConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelDataConfig,
    lambda data: ModelDataConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelProcessorConfig,
    lambda data: ModelProcessorConfig(
        traj_len=data["traj_len"],
        lead_len=data["lead_len"],
        output_dir=data["output_dir"],
        masking_strategy=MaskingStrategy[data["masking_strategy"]],
        masking_percentage=data["masking_percentage"],
        min_sog=data["min_sog"],
        max_time_gap=data["max_time_gap"],
        max_traj_gap_distance_m=data["max_traj_gap_distance_m"],
    )
)

ConfigVisitorRegistry.register(
    ModelTrainerConfig,
    lambda data: ModelTrainerConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelConfig,
    lambda data: ModelConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelLossConfig,
    lambda data: ModelLossConfig(**data)
)

ConfigVisitorRegistry.register(
    ModelDataUploadHandlerConfig,
    lambda data: ModelDataUploadHandlerConfig(**data)
)
