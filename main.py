from logger import logger
from pipeline.data_ingestion import DataIngestion
from pipeline.data_transformation import DataTransformation
logger.info(f"----------We are printing the logs here!!!".ljust(60, '-'))

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = DataIngestion()
    data = obj.initiateDataIngestion()
    
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation"

try:
    logger.info(f"----------Stage {STAGE_NAME} Started".ljust(60, '-'))
    obj = DataTransformation()
    obj.initiate_data_transformation(data)
    logger.info(f"----------Stage {STAGE_NAME} Completed".ljust(60, '-'))
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
