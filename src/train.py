from Config.parser import parse_config
from Connection.postgres_connection import PostgresConnection
from ForceData.force_data_access_handler_db import ForceDataAccessHandlerDb
from ForceProviders.force_provider_depth import DepthForceProvider
from Model.model import Model
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ModelPipeline.trainer import Trainer
from ModelUtils.data_loader import AisDataLoader
from ModelUtils.data_processor import DataProcessor
from ModelUtils.loss_calculator import LossCalculator


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    force_data_connection = PostgresConnection(cfg.postgresCfg)
    force_data_handler = ForceDataAccessHandlerDb(force_data_connection)
    force_provider_depth = DepthForceProvider(force_data_handler, cfg.depthForceProviderCfg)

    data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    data_loader = AisDataLoader(data_processor, cfg.modelDataLoaderCfg)
    train_data_loader, test_data_loader, stats = data_loader.get_data_loaders()
    loss_calculator = LossCalculator(cfg.modelLossCfg)
    model = Model(stats, [force_provider_depth], loss_calculator, cfg.modelCfg)

    model = model.to(cfg.modelCfg.device)

    trainer = Trainer(model, train_data_loader, test_data_loader, cfg.modelTrainerCfg)
    trainer.train()
