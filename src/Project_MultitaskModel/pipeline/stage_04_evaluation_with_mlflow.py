from Project_MultitaskModel.components.evaluation import Evaluation
from Project_MultitaskModel.configs.configurations import ConfigureManager
from Project_MultitaskModel import logger


STAGE_NAME = "Evaluation with MLflow"
class EvaluationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigureManager()
        eval_config = config.get_evaluation_model_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e