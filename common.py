import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from logger import logger
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import holidays
from datetime import timedelta, datetime

def add_lags(dff):
    try:
        target_map = dff['consumed_unit'].to_dict()
        # 15 minutes, 30 minutes, 1 hour
        dff['lag1'] = (dff.index - pd.Timedelta('15 minutes')).map(target_map)
        dff['lag2'] = (dff.index - pd.Timedelta('30 minutes')).map(target_map)
        dff['lag3'] = (dff.index - pd.Timedelta('1 day')).map(target_map)
        dff['lag4'] = (dff.index - pd.Timedelta('7 days')).map(target_map)
        dff['lag5'] = (dff.index - pd.Timedelta('15 days')).map(target_map)
        dff['lag6'] = (dff.index - pd.Timedelta('30 days')).map(target_map)
        dff['lag7'] = (dff.index - pd.Timedelta('45 days')).map(target_map)
        return dff
    
    except KeyError as e:
        print(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def create_features(hourly_data):
    hourly_data = hourly_data.copy()

    # Check if the index is in datetime format
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        hourly_data.index = pd.to_datetime(hourly_data.index)

    hourly_data['day'] = hourly_data.index.day
    hourly_data['hour'] = hourly_data.index.hour
    hourly_data['month'] = hourly_data.index.month
    hourly_data['dayofweek'] = hourly_data.index.dayofweek
    hourly_data['quarter'] = hourly_data.index.quarter
    hourly_data['dayofyear'] = hourly_data.index.dayofyear
    hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
    hourly_data['year'] = hourly_data.index.year
    return hourly_data

def data_validation_filtering(dfs):
    try:
        """ validation and filtering """
        # frequency and pf
        condition = (
            (dfs['Frequency'] > 51) | (dfs['Frequency'] < 49)  |
            (dfs['R_PF'] > 1) | (dfs['Y_PF'] > 1) | (dfs['B_PF'] > 1) |  # Check if any PF is greater than 1
            (dfs['R_PF'] < 0) | (dfs['Y_PF'] < 0) | (dfs['B_PF'] < 0)    # Check if any PF is less than 0
        )   
        # Apply the condition to set 'KWh' to NaN
        dfs.loc[condition, 'kWh'] = np.nan
        dfs.drop(['R_PF','Y_PF','B_PF','Frequency'],axis= 1 ,inplace= True)
        # voltage
        no_voltage_df = dfs[(dfs['R_Voltage'] == 0) & (dfs['Y_Voltage'] == 0) & (dfs['B_Voltage'] == 0)]
        if not no_voltage_df.empty:
            no_voltage_but_current = no_voltage_df[(no_voltage_df['R_Current'] != 0) & (no_voltage_df['B_Current'] != 0) & (no_voltage_df['Y_Current'] != 0)]

            if not no_voltage_but_current.empty:
                dfs.loc[no_voltage_but_current.index, 'kWh'] = np.nan
        # current
        no_current_df = dfs[(dfs['R_Current'] == 0) & (dfs['Y_Current'] == 0) & (dfs['B_Current'] == 0)]
        if not no_current_df.empty:
            load_with_no_current_df = no_current_df[(no_current_df['Load_kW']>0.03) & (no_current_df['Load_kVA']>0.03)]
            
            if not load_with_no_current_df.empty:
                    dfs.loc[load_with_no_current_df.index, 'kWh'] = np.nan
        dfs.drop(['R_Voltage','Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current','B_Current','Load_kW','Load_kVA'],axis= 1 ,inplace= True)
        return dfs
    except Exception as e:
        print("error in validation :",e,e.args())

def data_validation(collection,label_sensor_id,s_df):
    try:
        s_df["creation_time"] = pd.to_datetime(s_df['creation_time'])
        s_df.set_index(['creation_time'],drop= True, inplace= True)

        site_id = s_df['site_id'].unique()[0]
        # label_sensor_id = s_df['label_sensor_id'].unique()[0]
        if len(s_df) > 3000:
            description = s_df.describe()
            Q2 = description.loc['50%', 'opening_KWh']
            if Q2 < 1:
                return None
            # outage situation
            s_df.loc[s_df['opening_KWh'] == 0, "opening_KWh"] = np.nan
            s_df.loc[s_df['opening_KWh'].first_valid_index():]
            s_df.bfill(inplace=True)

            # missing packet
            sensor_df = s_df[['opening_KWh']].resample(rule="15min").asfreq()
            sensor_df = sensor_df.infer_objects(copy=False)
            sensor_df.interpolate(method="linear", inplace=True)

            # no consumption / same reading
            if sensor_df['opening_KWh'].nunique() < 10:
                return None

            # previous value of opening_KWh
            sensor_df['prev_KWh'] = sensor_df['opening_KWh'].shift(1)
            sensor_df.dropna(inplace=True)
            # if len(sensor_df[sensor_df['prev_KWh'] > sensor_df['opening_KWh']]) > 25:
            #     return None

            # consumed unit
            sensor_df['consumed_unit'] = sensor_df['opening_KWh'] - sensor_df['prev_KWh']
            sensor_df.loc[sensor_df['consumed_unit'] < 0, "opening_KWh"] = sensor_df["prev_KWh"]
            sensor_df.loc[sensor_df['consumed_unit'] < 0, "consumed_unit"] = 0

            if sensor_df['consumed_unit'].nunique() < 10:
                return None

            # eliminating id's based on slope
            numeric_index = pd.to_numeric(sensor_df.index)
            correlation = np.corrcoef(numeric_index, sensor_df['opening_KWh'])[0, 1]
            coeffs = np.polyfit(numeric_index, sensor_df['opening_KWh'], 1)

            slope = coeffs[0]
            if not np.abs(correlation) > 0.8 and slope > 0:
                return None

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
            sensor_df.drop(['opening_KWh', 'prev_KWh','db_outlier'],axis=1,inplace= True)
            # sensor_df.reset_index(inplace=True)
            dfresample = add_lags(sensor_df)
            dfresample = create_features(dfresample)
            start_date,end_date = dfresample.first_valid_index(), dfresample.last_valid_index()
            dfresample.reset_index(inplace=True)
            dfresample['label_sensor_id'] = label_sensor_id

            logger.info(f"sensor_id{label_sensor_id}done")
            # print("site_id:",site_id)
            if dfresample.empty is False:
                try:
                    # print(type(collection))
                    weather_data = data_from_weather_api(collection,site_id, start_date, end_date)
                    # logs_config.logger.info(f"length of weather_data:{len(weather_data)}")

                    if not weather_data.empty:
                        # print(f"weather_data:{len(weather_data)}")
                        weather_data['time'] = pd.to_datetime(weather_data['time'])
                        weather_data.set_index('time', inplace=True)

                        weather_data = weather_data[~weather_data.index.duplicated(keep='first')]
                        weather_data = weather_data.resample('15 min').ffill()

                        # Convert the creation_time columns to datetime if they are not already
                        weather_data.reset_index(inplace=True)
                        weather_data['creation_time'] = pd.to_datetime(weather_data['time'])
                        # df2['creation_time'] = pd.to_datetime(df2['creation_time'])
                        # return weather_data, df2
                        merged_df = weather_data.merge(dfresample, on='creation_time', how="inner")
                        # logger.info(f"columns in weather_data: {tuple(weather_data.columns)}")
                        # logger.info(f"columns in merged_df: {tuple(merged_df.columns)}")
                        # logger.info(f"columns in dfresample: {tuple(dfresample.columns)}")
                        merged_df.drop(['time', '_id', 'site_id','creation_time_iso'],axis=1, inplace= True)
                        # print(merged_df.head())
                        return merged_df
                    else:
                        print("weather_data not found")
                except Exception as e:
                    print(e)
            return None
    except Exception as e:
         print("error in kWh validation:",e,e.args)

# def holidays_list(start_date_str, end_date_str):
#     try:
#         start_date = start_date_str.date()
#         end_date = end_date_str.date()
#         holiday_list = []
#         india_holidays = holidays.CountryHoliday('India', years=start_date.year)
#         current_date = start_date
#         while current_date <= end_date:
#             if current_date in india_holidays or current_date.weekday() == 6:
#                 holiday_list.append(current_date)
#             current_date += timedelta(days=1)
#         return holiday_list
#     except Exception as e:
#         return None
    
def holidays_list(start_date, end_date):
    try:
        start_date = start_date.date()
        end_date = end_date.date()
        holiday_list = []
        india_holidays = holidays.CountryHoliday('India', years=range(start_date.year, end_date.year + 1))        
        current_date = start_date
        while current_date <= end_date:
            if current_date in india_holidays or current_date.weekday() == 6:  # Sunday is day 6
                holiday_list.append(current_date)
            current_date += timedelta(days=1)
        return holiday_list
    
    except Exception as e:
        print(f"Error in holidays_list: {e}")
        return None


def get_database(client,database_name):
    try:
        database = client[database_name] 
        collection = database['weather_data']
        return collection
    except Exception as e:
        print("error in fetching mongo database")
        raise e
        

def data_from_weather_api(collection,site, startDate, endDate):
    ''' Fetch weather data from CSV file based on date range'''
    # logger.info("Weather data fetching")
    try:
        start_date = startDate.strftime('%Y-%m-%d %H:%M:%S')
        end_date = endDate.strftime('%Y-%m-%d %H:%M:%S')
        # print(start_date,end_date)
        # conn = get_connection()
        # collection_name = os.getenv("weatherData")
        # loadProfile = db.weather_data
        documents = []
        query = collection.find({
            "site_id": site,
            "time": {
                "$gte": start_date,
                "$lte": end_date
            }
        })
        # print(query)
        for doc in query:
            documents.append(doc)
        try:
            df = pd.DataFrame(documents)
            # print(df.head())
            return df
        except Exception as e:
            print(e)
    except Exception as e:
        print("Error:", e)

def label_encoding_decoding(df):
    try:    
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
        with open('encoding_decoding_pair.json', 'w') as file:
            file.write(json_string)
        # df.drop('sensor_id',axis=1 ,inplace=True)
        return df
    except Exception as e:
        logger.error("error in encoding_decoding: {e}")