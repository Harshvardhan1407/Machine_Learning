from logger import logger
from pipeline.data_ingestion import DataIngestion
from pipeline.data_transformation import DataTransformation
from pipeline.model_training import ModelTraining
from pipeline.model_evaluation import ModelEvaluation
logger.info(f"----------We are printing the logs here!!!".ljust(60, '-'))

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = DataIngestion()
    data = obj.initiateDataIngestion() 
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))

except Exception as e:
    logger.info("error at ingestion stage")

STAGE_NAME = "Data Transformation"
try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = DataTransformation()
    tranformed_data = obj.initiate_data_transformation(data)
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))
except Exception as e:
    logger.info("error at transformation stage")


STAGE_NAME = "Model Training"

try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = ModelTraining()
    obj.initiate_model_training(tranformed_data)
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))
except Exception as e:
    logger.info("error at model training stage")


STAGE_NAME = "Model evaluation"

try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = ModelEvaluation()
    obj.initiate_model_evaluation(tranformed_data)
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))
except Exception as e:
    logger.info("error at model evaluation stage")
