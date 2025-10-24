from Config.parser import parse_config
from DataAccess.data_access_handler import DataAccessHandler
from ForceProviders.force_provider_depth import DepthForceProvider
from ForceProviders.force_provider_propulsion import PropulsionForceProvider
from ForceProviders.force_provider_traffic import TrafficForceProvider
from Utils.heatmap_generator import generate_heatmap_image
from DataAccess.postgres_connection import PostgresConnection
from params import Params


def main():
    cfg = parse_config("config.yaml")

    # prov_traffic = TrafficForceProvider(DataAccessHandler(
    #     PostgresConnection(cfg.postgresCfg)), cfg.trafficForceProviderCfg)

    # generate_heatmap_image(prov_traffic._vectormap, cfg.heatmapGeneratorCfg)

    # prov_depth = DepthForceProvider(DataAccessHandler(PostgresConnection(cfg.postgresCfg)), cfg.depthForceProviderCfg)
    # generate_heatmap_image(prov_depth._vectormap, cfg.heatmapGeneratorCfg)

    prov_propulsion = PropulsionForceProvider()

    prov_propulsion.get_force(Params(lat=59.3293, lon=18.0686, sog=5.0, cog=123.4, time=1625078400))


if __name__ == "__main__":
    main()
