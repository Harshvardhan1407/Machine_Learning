try:
    from logger import logger
    from config import problem_type, kind
    import pandas as pd
    import numpy as np
    import os
    import pickle
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
                
            if problem_type == "classification":
                return image_data_transformation(data)
        except Exception as e:
            logger.error(f"error in transformation :{e}",exc_info=True)


###########################----LOAD FORECASTING----#########################################################################################
def load_forecasting_data_transforamtion(data):
    try:
        logger.info(f"----------load profile data transformation".ljust(60, '-'))
        df = data.copy()
        logger.info(f"object type columns:{[col for col in df.columns if type(df[col]) == object]}".ljust(60, '-')) 
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
