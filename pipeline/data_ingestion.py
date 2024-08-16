import pandas as pd
import os
from logger import logger
from config import data_path

class DataIngestion:
    def initiateDataIngestion(self):
        try:
            file_name, file_extension = os.path.splitext(data_path)
            if file_extension == ".csv":
                if file_name == r"Datasets\flight_data":
                    data = pd.read_csv(data_path,index_col=0) 
                else:
                    data = pd.read_csv(data_path)

            if file_extension == ".parquet":
                data = pd.read_parquet(data_path)
            
            if file_extension == ".json":
                data = pd.read_json(data_path)

            logger.info(f"----------data read".ljust(60, '-'))
            logger.info(f"----------columns:{tuple(data.columns)}".ljust(60, '-'))
            logger.info(f"----------shape:{data.shape}".ljust(60, '-'))
            return data
        
        except Exception as e:
            logger.error(f"Error in Ingestion Process: {e}",exc_info=True)