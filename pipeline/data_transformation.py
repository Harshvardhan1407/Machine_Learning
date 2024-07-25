try:
    from logger import logger
    from config import problem_type
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit
    import category_encoders as ce
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    pd.set_option('display.max_columns', None)

except ImportError as e:
    print("ImportError in data_transformation:",e)

class DataTransformation:
    def initiate_data_transformation(self,data):
        try:
            logger.info(f"----------transformation start".ljust(60, '-'))
            # logger.info("------transformation start------")
            if problem_type == "regression":
                return flight_data_transformation(data)
            if problem_type == "classification":
                return image_data_transformation(data)
            logger.info(f"----------transformation done".ljust(60, '-'))
        except Exception as e:
            logger.info("error in transformation :",e)

def flight_data_transformation(df):
    try:
        logger.info(f"----------flight data transformation".ljust(60, '-'))
        # conversion for all categorical features
        value_to_map= {}
        for col in df.columns:
            # print(col)
            if (df[col].dtype== object) and (col not in ["flight","price"]):
                i=0
                for index in df[col].unique():
                    if not index in value_to_map.keys():
                        value_to_map[index] = i
                        i+=1
                df[col] = df[col].map(value_to_map).astype(int)

        # flight column conversion
        label_encoder = LabelEncoder()
        df['flight'] = label_encoder.fit_transform(df['flight'])
        df['flight_decoded'] = label_encoder.inverse_transform(df['flight'])
        json_data = df[['flight',"flight_decoded"]].to_json(orient='records',lines = True)

        # Save the JSON data to a file
        with open('encoding_decoding_pair.json', 'w') as file:
            file.write(json_data)

        print(df.head())
    except Exception as e:
        logger.info("error in flight data transformation :",e)

def image_data_transformation(data):
    try:
        # logger.info("------flight data transformation------")
        logger.info(f"----------flight data transformation".ljust(60, '-'))

    except Exception as e:
        logger.info("error in flight data transformation :",e)