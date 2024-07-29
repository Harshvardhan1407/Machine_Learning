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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import cross_val_score

    pd.set_option('display.max_columns', None)

except ImportError as e:
    print("ImportError in model training:", e)

class ModelTraining():
    pass
    def initiate_model_training(self,df):
        try:
            train_set, test_set = train_test_split(df, test_size=0.001, random_state=42)
            logger.info(f"---------- train {len(train_set)}".ljust(60, '-'))
            logger.info(f"---------- test {len(test_set)}".ljust(60, '-'))
            train_set.reset_index(drop=True, inplace= True)
            X_train = train_set.drop(['price'],axis=1)
            y_train = train_set['price']
            # linear regreesion
            lin_reg = LinearRegression()
            lin_reg.fit(X_train, y_train)
            # logger.info(f"---------- data transformation".ljust(60, '-'))
            tree_reg = DecisionTreeRegressor()
            tree_reg.fit(X_train, y_train)

            # scores = cross_val_score(tree_reg, X_test, y_test,
            # scoring="neg_mean_squared_error", cv=10)
            # tree_rmse_scores = np.sqrt(-scores)
            # forest_reg = RandomForestRegressor()
            # forest_reg.fit(X_train, y_train)


            def testing(self):
                test_set.reset_index(drop= True, inplace= True)
                X_test = test_set.drop(['price'],axis= 1)
                y_test = test_set['price']

            # def prediction(self):
            #     price_pred = lin_reg.predict(X_test)

        except Exception as e:
            logger.info(f"----------error in model training".ljust(60, '-'))
