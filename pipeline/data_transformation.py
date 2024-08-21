try:
    from logger import logger
    from config import problem_type, kind
    import pandas as pd
    import numpy as np
    import os
    import pickle
    import json
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()
    import holidays

    from sklearn.cluster import DBSCAN
    from common import add_lags, create_features, get_database, holidays_list
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import StratifiedShuffleSplit
    import category_encoders as ce
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from concurrent.futures import ThreadPoolExecutor
    import os
    import concurrent
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from bson import ObjectId

    pd.set_option('display.max_columns', None)

except ImportError as e:
    print("ImportError in data_transformation:",e)

class DataTransformation:
    def initiate_data_transformation(self,data):
        try:
            if problem_type == "regression":
                if kind == "flight":
                    return flight_data_transformation(data)
                if kind == "load_forecasting":
                    return load_forecasting_data_transforamtion(data)
                if kind == "load_forecasting2":
                    return parquet_data_transforamtion(data)
                
            if problem_type == "classification":
                return image_data_transformation(data)
        except Exception as e:
            logger.error(f"error in transformation :{e}",exc_info=True)


###########################----LOAD FORECASTING----#########################################################################################
def load_forecasting_data_transforamtion(data):
    try:
        logger.info(f"----------load profile data transformation".ljust(60, '-'))
        df = data.copy()
        logger.info(f"object type columns:{[col for col in df.columns if df[col].dtype == object]}".ljust(60, '-')) 
        df["Date_Time"] = pd.to_datetime(df['Date_Time'],dayfirst=True)
        df.set_index(['Date_Time'],drop= True, inplace= True)
        # checks 'flat_id','kWh','R_Voltage','Y_Voltage','B_Voltage','R_Current','Y_Current','B_Current','R_PF', 'Y_PF','B_PF', 'Frequency', 'Load_kW', 'Load _kVA', 'Date_Time')

        if len(df.loc[(df['Frequency'] > 51) | (df['Frequency'] < 49)]) !=0 :
            logger.info(f"Frequency disturbance: {len(df.loc[(df['Frequency'] > 51) | (df['Frequency'] < 49)])}")

        # if len(df.loc[(df['Frequency'] > 51) | (df['Frequency'] < 49)]) !=0 :
        #     logger.info(f"Frequency disturbance: {len(df.loc[(df['Frequency'] > 51) | (df['Frequency'] < 49)])}")
        # columns_to_consider = ['Date_Time']

        # corr_matrix = df.corr()
        # # Plot the correlation matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Matrix')
        # plt.show()
        # return df

    except Exception as e:
        logger.error(f"error in load profile data transformation:{e}",exc_info=True)

###########################----PARQUET DATA ----#########################################################################################
def parquet_data_transforamtion(data):
    try:
        logger.info(f"----------parquet data transformation".ljust(60, '-'))
        df = data.copy()
        df.drop(['_id'],axis=1, inplace=True)
        # logger.info(f"object type columns:{[col for col in df.columns if df[col].dtype == object]}".ljust(60, '-')) 

        # date time handling and index creation
        df["creation_time"] = pd.to_datetime(df['creation_time'])
        df.set_index(['creation_time'],drop= True, inplace= True)
        df.sort_index(inplace= True)

        # sensor_id_list = df['sensor_id'].unique()

        start_date, end_date = df.first_valid_index(), df.last_valid_index()

        # Generate the holiday list
        holiday = holidays_list(start_date, end_date)
        # Convert the holiday list to a pandas DatetimeIndex
        if holiday is not None:
            holiday_dates = pd.to_datetime(holiday)
            df['is_holiday'] = df.index.isin(holiday_dates).astype(int)
        else:
            logger.info("Holiday list generation failed.")
        df.reset_index(inplace=True)
        site_df = pd.read_json(r"paired_sensor_site_data.json")
        df_merged = df.merge(site_df, on='sensor_id', how='left')

        logger.info(f"columns: {df_merged.columns}")
        df_merged["creation_time"] = pd.to_datetime(df_merged['creation_time'])
        df_merged.set_index(['creation_time'],drop= True, inplace= True)
        print(df_merged.info())


    except Exception as e:
        logger.error(f"error in load profile data transformation:{e}",exc_info=True)
###########################----FLIGHT DATA----#########################################################################################

def flight_data_transformation(df):
    try:
        logger.info(f"----------flight data transformation".ljust(60, '-'))
        # conversion for all categorical features
        value_to_map= {}
        for col in df.columns:
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
        df.drop('flight_decoded',axis=1 ,inplace=True)
        standardscaler = StandardScaler()
        df["standardized_flight"] = standardscaler.fit_transform(df[['flight']])
        minmaxscaler = MinMaxScaler()
        df["normalized_flight"] = minmaxscaler.fit_transform(df[['flight']])
        corr_matrix = df.corr()
        # Plot the correlation matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Matrix')
        # plt.show()

        return df

    except Exception as e:
        logger.error(f"error in flight data transformation:{e}",exc_info=True)

###########################----IMAGE CLASSIFICATION----#########################################################################################

def image_data_transformation(data):
    try:
        logger.info(f"----------flight data transformation".ljust(60, '-'))

    except Exception as e:
        logger.error(f"error in image data transformation:{e}",exc_info=True)

###########################---- ----#########################################################################################

























# import mongo_query.sensor_ids as sensor_ids
# from config.db_port import get_database

client = MongoClient(os.getenv("mongo_url"),compressors='zstd',zlibCompressionLevel=9)
database_name = os.getenv("database_name")
db = get_database(client,database_name)
sensor = db.load_profile_jdvvnl

def data_fetch(sensor_id, site_id):
    try:
        from_id = f"{sensor_id}-2024-01-01 00:00:00"
        to_id = f"{sensor_id}-2024-03-31 23:59:59"
        query = {"_id": {"$gte": from_id, "$lt": to_id}}

        results = list(sensor.find(query))
        for doc in results:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])

        if results:
            df = datatransformation.init_transformation(results, site_id)
            if df is None:
                logger.info(f"Nothing transformed for sensor_id: {sensor_id}")
            else:
                logger.info(f"Fetched and transformed {len(df)} records for sensor_id: {sensor_id}")
            return df
        else:
            logger.info(f"No records found for sensor_id: {sensor_id}")
            return None

    except Exception as e:
        logger.error(f"Error fetching data for sensor_id {sensor_id}: {e}", exc_info=True)
        return None

def fetch_data_for_sensors(circle_id, output_dir="sensor_data"):
    os.makedirs(output_dir, exist_ok=True)

    sensors = sensor_ids.sensor_ids(circle_id)
    sensorids = [doc["id"] for doc in sensors]
    site_ids = [doc["site_id"] for doc in sensors]

    all_dicts = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(data_fetch, sensor_id, site_id): (sensor_id, site_id) for sensor_id, site_id in zip(sensorids, site_ids)}

        for future in concurrent.futures.as_completed(futures):
            dicts = future.result()
            if dicts is not None and len(dicts) > 0:
                all_dicts.extend(dicts)

    if all_dicts:
        combined_df = pd.DataFrame(all_dicts)
        
        # Convert all object type columns to strings if necessary
        combined_df = combined_df.applymap(lambda x: str(x) if isinstance(x, ObjectId) else x)

        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, os.path.join(output_dir, f"{circle_id}_data.parquet"))
        
        logger.info(f"Saved data for circle_id: {circle_id}")
        return "saved"
    else:
        logger.info(f"No data to save for circle_id: {circle_id}")
        return "no data"

