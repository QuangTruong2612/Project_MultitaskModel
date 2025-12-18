from src.Project_MultitaskModel.components.prepare_base_model import PrepareBaseModel
from src.Project_MultitaskModel.configs.configurations import ConfigureManager
from src.Project_MultitaskModel import logger

STAGE_NAME = "Prepare Base Model Stage"
class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigureManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e