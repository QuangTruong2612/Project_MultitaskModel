from src.Project_MultitaskModel.components.training_model import TrainingModel
from src.Project_MultitaskModel.configs.configurations import ConfigureManager
from src.Project_MultitaskModel import logger


STAGE_NAME = "Training Model Stage"
class TrainingModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigureManager()
        training_model_config = config.get_training_model_config()
        trainer = TrainingModel(config=training_model_config)
        trainer.train_model()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = TrainingModelPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e