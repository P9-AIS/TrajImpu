from Config.parser import parse_config
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceProviders.force_provider_depth import DepthForceProvider
from ForceProviders.force_provider_propulsion import PropulsionForceProvider
from ForceProviders.force_provider_traffic import TrafficForceProvider
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ForceUtils.heatmap_generator import generate_heatmap_image
from Connection.postgres_connection import PostgresConnection
from Types.params import Params
# from ModelUtils.data_processor import DataProcessor
from ModelUtils.data_loader import AisDataLoader


def main():
    cfg = parse_config("config.yaml")

    # prov_traffic = TrafficForceProvider(ForceDataAccessHandlerDb(
    #     PostgresConnection(cfg.postgresCfg)), cfg.trafficForceProviderCfg)

    # generate_heatmap_image(prov_traffic.get_vectormap(), cfg.heatmapGeneratorCfg)
    data_handler = ModelDataAccessHandlerCSV(cfg.modelDatasetCfg)
    # data_handler.get_ais_messages()
    train_loader, test_loader = AisDataLoader.get_data_loaders(cfg.modelDataLoaderCfg, data_handler)
    print("well well well...")
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
