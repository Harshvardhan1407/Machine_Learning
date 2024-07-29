import pandas as pd
from logger import logger
from config import data_path
class DataIngestion:
    def initiateDataIngestion(self):
        try:
            data = pd.read_csv(data_path,index_col=0) 
            # logger.info("data read")
            logger.info(f"----------data read".ljust(60, '-'))

            return data
        
        except Exception as e:
            logger.info(f"Error in Ingestion Process: {e}")