from logger import logger
from pipeline.data_ingestion import DataIngestion
from pipeline.data_transformation import DataTransformation
logger.info("We are printing the logs here!!!")

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"-----------Stage {STAGE_NAME} Started------------")
    obj = DataIngestion()
    data = obj.initiateDataIngestion()
    
    logger.info(f"-----------Stage {STAGE_NAME} Completed-----------")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation"

try:
    logger.info(f"-----------Stage {STAGE_NAME} Started------------")
    obj = DataTransformation()
    obj.initiate_data_transformation(data)
    logger.info(f"-----------Stage {STAGE_NAME} Completed------------")
except Exception as e:
    logger.exception(e)
    raise e

# STAGE_NAME = "Model Training"

# try:
#     logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
#     obj = ModelTrainingPipeline()
#     obj.main()
#     logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Model evaluation stage"
# try:
#     logger.info("-----------Stage {} Started------------".format(STAGE_NAME))
#     obj = ModelEvaluationTrainingPipeline()
#     obj.main()
#     logger.info("-----------Stage {} Completed------------".format(STAGE_NAME))
# except Exception as e:
#     logger.exception(e)
#     raise e
