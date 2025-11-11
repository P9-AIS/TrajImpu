from Config.parser import parse_config
from Model.model import Model
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ModelPipeline.trainer import Trainer
from ModelUtils.data_loader import AisDataLoader
from ModelUtils.data_processor import DataProcessor


if __name__ == "__main__":
    cfg = parse_config("config.yaml")

    data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    data_loader = AisDataLoader(data_processor, cfg.modelDataLoaderCfg)
    train_data_loader, test_data_loader = data_loader.get_data_loaders()
    model = Model(cfg.modelCfg)

    trainer = Trainer(model, train_data_loader, test_data_loader, cfg.modelTrainerCfg)
    trainer.train()
