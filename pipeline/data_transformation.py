try:
    from logger import logger
    from config import problem_type
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
            # logger.info(f"----------transformation start".ljust(60, '-'))
            # logger.info("------transformation start------")
            if problem_type == "regression":
                return flight_data_transformation(data)
            if problem_type == "classification":
                return image_data_transformation(data)
            # logger.info(f"----------transformation done".ljust(60, '-'))
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

        # print(df.head())
    except Exception as e:
        logger.info("error in flight data transformation :",e)

def image_data_transformation(data):
    try:
        # logger.info("------flight data transformation------")
        logger.info(f"----------flight data transformation".ljust(60, '-'))

    except Exception as e:
        logger.info("error in flight data transformation :",e)


########################################################################################################





# try:
#     from logger import logger
#     from config import problem_type
#     import pandas as pd
#     import numpy as np
#     import os
#     import pickle
#     import matplotlib.pyplot as plt
#     from sklearn.model_selection import train_test_split
#     from sklearn.model_selection import StratifiedShuffleSplit
#     import category_encoders as ce
#     from sklearn.preprocessing import LabelEncoder
#     import seaborn as sns
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#     from sklearn.base import BaseEstimator, TransformerMixin
#     from sklearn.compose import ColumnTransformer
#     from sklearn.pipeline import Pipeline, FeatureUnion
#     pd.set_option('display.max_columns', None)

# except ImportError as e:
#     print("ImportError in data_transformation:", e)

# class CategoricalMapper(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.value_to_map = {}
    
#     def fit(self, X, y=None):
#         for col in X.columns:
#             if X[col].dtype == object and col not in ["flight", "price"]:
#                 i = 0
#                 self.value_to_map[col] = {}
#                 for index in X[col].unique():
#                     if index not in self.value_to_map[col].keys():
#                         self.value_to_map[col][index] = i
#                         i += 1
#         return self
    
#     def transform(self, X, y=None):
#         X = X.copy()
#         for col in self.value_to_map:
#             X[col] = X[col].map(self.value_to_map[col]).astype(int)
#         return X

# class SaveAndEncodeFlight(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.label_encoder = LabelEncoder()
    
#     def fit(self, X, y=None):
#         self.label_encoder.fit(X['flight'])
#         return self
    
#     def transform(self, X, y=None):
#         X = X.copy()
#         X['flight'] = self.label_encoder.transform(X['flight'])
#         X['flight_decoded'] = self.label_encoder.inverse_transform(X['flight'])
#         json_data = X[['flight', 'flight_decoded']].to_json(orient='records', lines=True)
        
#         with open('encoding_decoding_pair.json', 'w') as file:
#             file.write(json_data)
        
#         X.drop('flight_decoded', axis=1, inplace=True)
#         return X

# class DataTransformation:
#     def initiate_data_transformation(self, data):
#         try:
#             if problem_type == "regression":
#                 return self.flight_data_transformation(data)
#             if problem_type == "classification":
#                 return self.image_data_transformation(data)
#         except Exception as e:
#             logger.info("error in transformation:", e)
    
#     def flight_data_transformation(self, df):
#         try:
#             logger.info(f"----------flight data transformation".ljust(60, '-'))
            
#             # Instantiate your transformers
#             categorical_mapper = CategoricalMapper()
#             save_and_encode_flight = SaveAndEncodeFlight()
#             standard_scaler = StandardScaler()
#             min_max_scaler = MinMaxScaler()

#             # Create the pipeline
#             preprocessing_pipeline = Pipeline([
#                 ('categorical_mapper', categorical_mapper),
#                 ('save_and_encode_flight', save_and_encode_flight),
#                 ('feature_scaling', FeatureUnion(transformer_list=[
#                     ('standardized_flight', Pipeline([
#                         ('extract_flight', ColumnTransformer([('flight', 'passthrough', ['flight'])])),
#                         ('scaler', StandardScaler())
#                     ])),
#                     ('normalized_flight', Pipeline([
#                         ('extract_flight', ColumnTransformer([('flight', 'passthrough', ['flight'])])),
#                         ('scaler', MinMaxScaler())
#                     ]))
#                 ]))
#             ])

#             # Fit and transform the dataframe
#             df_transformed = preprocessing_pipeline.fit_transform(df)

#             # Convert the transformed array back to DataFrame for visualization and further use
#             df_transformed = pd.DataFrame(df_transformed, columns=['standardized_flight', 'normalized_flight'])

#             # Visualize the correlation matrix
#             corr_matrix = df_transformed.corr()
#             plt.figure(figsize=(10, 8))
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#             plt.title('Correlation Matrix')
#             plt.show()

#             # Save the pipeline
#             with open('preprocessing_pipeline.pkl', 'wb') as file:
#                 pickle.dump(preprocessing_pipeline, file)

#             return df_transformed
#         except Exception as e:
#             logger.info("error in flight data transformation:", e)
    
#     def image_data_transformation(self, data):
#         try:
#             logger.info(f"----------image data transformation".ljust(60, '-'))
#             # Implement image data transformation logic here
#         except Exception as e:
#             logger.info("error in image data transformation:", e)

