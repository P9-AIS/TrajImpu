from Config.parser import parse_config
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceProviders.force_provider_depth import DepthForceProvider
from ForceProviders.force_provider_propulsion import PropulsionForceProvider
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ForceUtils.heatmap_generator import generate_heatmap_image
from Connection.postgres_connection import PostgresConnection
from ForceTypes.params import Params
from ModelUtils.data_loader import AisDataLoader
from ModelUtils.data_processor import DataProcessor
from ForceUtils.geo_converter import GeoConverter as gc


def main():
    cfg = parse_config("config.yaml")

    area = cfg.depthForceProviderCfg.area

    print(gc.epsg3034_to_espg4326(area.bottom_left.E, area.bottom_left.N))
    print(gc.epsg3034_to_espg4326(area.top_right.E, area.top_right.N))

    # # prov_traffic = TrafficForceProvider(ForceDataAccessHandlerDb(
    # #     PostgresConnection(cfg.postgresCfg)), cfg.trafficForceProviderCfg)

    # # generate_heatmap_image(prov_traffic.get_vectormap(), cfg.heatmapGeneratorCfg)

    # data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    # data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    # train_loader, test_loader = AisDataLoader.get_data_loaders(cfg.modelDataLoaderCfg, data_processor)
    # print("well well well...")

    # prov_depth = DepthForceProvider(DataAccessHandler(PostgresConnection(cfg.postgresCfg)), cfg.depthForceProviderCfg)
    # generate_heatmap_image(prov_depth.get_vectormap(), cfg.heatmapGeneratorCfg)

    # ais_messages = ModelDataAccessHandlerCSV(cfg.modelDatasetCfg).get_ais_messages()
    # raw_ais = DataProcessor(cfg.modelDataProcessorCfg).raw_ais_to_dataset(ais_messages)
    # processed = DataProcessor(cfg.modelDataProcessorCfg).raw_ais_dataset_to_processed(raw_ais)
    # test_loader = DataLoader(
    #     processed,
    #     batch,
    #     shuffle
    # )

    # train_loader = DataLoader(
    #     processed,
    #     batch,
    #     shuffle
    # )


if __name__ == "__main__":
    main()
