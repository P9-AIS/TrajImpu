from Config.parser import parse_config
from DataAccess.data_access_handler import DataAccessHandler
from ForceProviders.force_provider_depth import DepthForceProvider
from ForceProviders.force_provider_traffic import TrafficForceProvider
from Utils.heatmap_generator import generate_heatmap_image
from DataAccess.postgres_connection import PostgresConnection


def main():
    cfg = parse_config("config.yaml")

    # prov_traffic = TrafficForceProvider(DataAccessHandler(
    #     PostgresConnection(cfg.postgresCfg)), cfg.trafficForceProviderCfg)
    # generate_heatmap_image(prov_traffic._vectormap, cfg.heatmapGeneratorCfg)

    prov_depth = DepthForceProvider(DataAccessHandler(PostgresConnection(cfg.postgresCfg)), cfg.depthForceProviderCfg)
    generate_heatmap_image(prov_depth._vectormap, cfg.heatmapGeneratorCfg)


if __name__ == "__main__":
    main()
