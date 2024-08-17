try:
    from logger import logger
    from config import problem_type, kind
    import pandas as pd
    import numpy as np
    import os
    import pickle
    import json
    from sklearn.cluster import DBSCAN
    from common import add_lags, create_features
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
        logger.info(f"object type columns:{[col for col in df.columns if df[col].dtype == object]}".ljust(60, '-')) 
        df["creation_time"] = pd.to_datetime(df['creation_time'])
        df.set_index(['creation_time'],drop= True, inplace= True)
        
        # sensor_id column conversion
        label_encoder = LabelEncoder()
        df['label_sensor_id'] = label_encoder.fit_transform(df['sensor_id'])
        
        sensor_ids = tuple(df['sensor_id'].unique())
        label_sensor_ids = tuple(df['label_sensor_id'].unique())
        json_data = [
            {str(sensor_id): str(label_sensor_id)}
            for sensor_id, label_sensor_id in zip(sensor_ids, label_sensor_ids)
        ]
        json_string = json.dumps(json_data)
        # print(1)
        # json_string = json.dumps(json_data)
        # Save the JSON data to a file
        with open('encoding_decoding_pair.json', 'w') as file:
            file.write(json_string)
        df.drop('sensor_id',axis=1 ,inplace=True)
        logger.info(f"columns{df.columns}")

        clean_data = []
        for s_id, data in df.groupby('label_sensor_id'):
            s_df = data.copy()
            
            if len(s_df) > 3000:
                description = s_df.describe()
                Q2 = description.loc['50%', 'opening_KWh']
                if Q2 < 1:
                    continue
                # outage situation
                s_df.loc[s_df['opening_KWh'] == 0, "opening_KWh"] = np.nan
                s_df.loc[s_df['opening_KWh'].first_valid_index():]
                s_df.bfill(inplace=True)

                # missing packet
                sensor_df = s_df.resample(rule="15min").asfreq()
                sensor_df.interpolate(method="linear", inplace=True)

                # no consumption / same reading
                if sensor_df['opening_KWh'].nunique() < 10:
                    continue

                # previous value of opening_KWh
                sensor_df['prev_KWh'] = sensor_df['opening_KWh'].shift(1)
                sensor_df.dropna(inplace=True)
                if len(sensor_df[sensor_df['prev_KWh'] > sensor_df['opening_KWh']]) > 25:
                    continue

                # consumed unit
                sensor_df['consumed_unit'] = sensor_df['opening_KWh'] - sensor_df['prev_KWh']
                sensor_df.loc[sensor_df['consumed_unit'] < 0, "opening_KWh"] = sensor_df["prev_KWh"]
                sensor_df.loc[sensor_df['consumed_unit'] < 0, "consumed_unit"] = 0

                if sensor_df['consumed_unit'].nunique() < 10:
                    continue

                # eliminating id's based on slope
                numeric_index = pd.to_numeric(sensor_df.index)
                correlation = np.corrcoef(numeric_index, sensor_df['opening_KWh'])[0, 1]
                coeffs = np.polyfit(numeric_index, sensor_df['opening_KWh'], 1)

                slope = coeffs[0]
                if not np.abs(correlation) > 0.8 and slope > 0:
                    continue

                # outlier detection
                epsilon = 11
                min_samples = 3
                dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
                # outlier_labels = dbscan.fit_predict(s_df[['consumed_unit']])
                # s_df['outlier'] = outlier_labels
                sensor_df['db_outlier'] = dbscan.fit_predict(sensor_df[['consumed_unit']])

                # print('no. of outliers were:',len(s_df[s_df['db_outlier']==-1]))
                sensor_df.loc[sensor_df['db_outlier'] == -1, 'consumed_unit'] = np.nan
                #  s_df.loc[outlier_dict[indices[0]].index,'consumed_unit'] = np.nan
                sensor_df.bfill(inplace=True)

                # sensor_df.reset_index(inplace=True)
                dfresample = add_lags(sensor_df)
                dfresample = create_features(dfresample)
                dfresample.reset_index(inplace=True)
                logger.info(f"sensor_id{s_id}done")
                clean_data.append(dfresample)
        # corr_matrix = df.corr()
        # # Plot the correlation matrix
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        # plt.title('Correlation Matrix')
        # plt.show()
        return clean_data

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
