from logger import logger

class DataTransformation:
    def initiate_data_transformation(self):
        try:
            logger.info("transformation start")
            pass
        except Exception as e:
            logger.info("error in transformation :",e)