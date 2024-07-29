try:
    from logger import logger
    from config import problem_type
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pandas as pd
    import numpy as np
    import os
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    def initiate_model_training(self,df):
        try:
            df.reset_index(drop=True, inplace= True)
            X = df.drop(['price'],axis=1)
            y = df['price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
            # train_set, test_set = train_test_split(df, test_size=0.001, random_state=42)
            logger.info(f"---------- X_train {len(X_train),X_train.shape}".ljust(60, '-'))
            logger.info(f"---------- X_test {len(X_test),X_test.shape}".ljust(60, '-'))
            logger.info(f"---------- y_train {len(y_train),y_train.shape}".ljust(60, '-'))
            logger.info(f"---------- y_test {len(y_test),y_test.shape}".ljust(60, '-'))

            # # linear regreesion
            # lin_reg = LinearRegression()
            # lin_reg.fit(X_train, y_train)
            # # logger.info(f"---------- data transformation".ljust(60, '-'))
            # tree_reg = DecisionTreeRegressor()
            # tree_reg.fit(X_train, y_train)

            # scores = cross_val_score(tree_reg, X_test, y_test,
            # scoring="neg_mean_squared_error", cv=10)
            # tree_rmse_scores = np.sqrt(-scores)
            # forest_reg = RandomForestRegressor()
            # forest_reg.fit(X_train, y_train)
           
            param_grid_lr = {
                'fit_intercept': [True, False]
            }

            param_grid_dtr = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10]
            }

            param_grid_rfr = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10]
            }

            # Initialize the models
            lr = LinearRegression()
            grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5, scoring='r2')
            # print("grid_search_lr",grid_search_lr)
            grid_search_lr.fit(X_train, y_train)
            best_lr = grid_search_lr.best_estimator_
            print("best_lr",best_lr)
            y_pred_lr = best_lr.predict(X_test)
            # Calculate metrics
            mae_lr = mean_absolute_error(y_test, y_pred_lr)
            mse_lr = mean_squared_error(y_test, y_pred_lr)
            r2_lr = r2_score(y_test, y_pred_lr)
            print(f"Linear Regression: MAE={mae_lr}, MSE={mse_lr}, R2={r2_lr}")




            dtr = DecisionTreeRegressor(random_state=42)
            grid_search_dtr = GridSearchCV(estimator=dtr, param_grid=param_grid_dtr, cv=5, scoring='r2')
            # print("grid_search_dtr",grid_search_dtr)
            grid_search_dtr.fit(X_train, y_train)
            best_dtr = grid_search_dtr.best_estimator_
            print("best_dtr",best_dtr)
            y_pred_dtr = best_dtr.predict(X_test)

            mae_dtr = mean_absolute_error(y_test, y_pred_dtr)
            mse_dtr = mean_squared_error(y_test, y_pred_dtr)
            r2_dtr = r2_score(y_test, y_pred_dtr)
            print(f"Decision Tree Regressor: MAE={mae_dtr}, MSE={mse_dtr}, R2={r2_dtr}")


            rfr = RandomForestRegressor(random_state=42)
            # Initialize GridSearchCV
            grid_search_rfr = GridSearchCV(estimator=rfr, param_grid=param_grid_rfr, cv=5, scoring='r2')
            # print("grid_search_rfr",grid_search_rfr)
            # Fit the models
            grid_search_rfr.fit(X_train, y_train)
            # Get the best estimators
            best_rfr = grid_search_rfr.best_estimator_
            print("best_rfr",best_rfr)
            # Evaluate the models
            y_pred_rfr = best_rfr.predict(X_test)

            
            mae_rfr = mean_absolute_error(y_test, y_pred_rfr)
            mse_rfr = mean_squared_error(y_test, y_pred_rfr)
            r2_rfr = r2_score(y_test, y_pred_rfr)

            print(f"Random Forest Regressor: MAE={mae_rfr}, MSE={mse_rfr}, R2={r2_rfr}")

            # def prediction(self):
            #     price_pred = lin_reg.predict(X_test)

        except Exception as e:
            logger.info(f"----------error in model training".ljust(60, '-'))
            print(e)
