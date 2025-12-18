import sys
import torch
from Config.parser import parse_config
from Model.simple import Model
from ModelData.model_data_access_handler_csv import ModelDataAccessHandlerCSV
from ModelUtils.data_loader import AisDataLoader
from ModelUtils.data_processor import DataProcessor
from ModelUtils.loss_calculator import LossCalculator
from ModelData.model_data_upload_handler_http import ModelDataUploadHandlerHTTP
from ModelPipeline.predicter import Predicter

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_simple <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    cfg = parse_config("config.yaml")

    upload_handler = ModelDataUploadHandlerHTTP(cfg.modelDataUploadHandlerCfg)

    data_handler = ModelDataAccessHandlerCSV(cfg.modelDataCfg)
    data_processor = DataProcessor(data_handler, cfg.modelDataProcessorCfg)
    data_loader = AisDataLoader(data_processor, cfg.modelDataLoaderCfg)
    _, _, test_data_loader, stats = data_loader.get_data_loaders()

    loss_calculator = LossCalculator()

    # Initialize model
    model = Model(stats, loss_calculator, cfg.simpleCfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use Predicter
    predicter = Predicter(model, upload_handler, test_data_loader, cfg.modelDataProcessorCfg, cfg.modelPredicterCfg)
    predicter.predict()
