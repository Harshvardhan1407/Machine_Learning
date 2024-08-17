import pandas as pd
def add_lags(df):
    try:
        target_map = df['consumed_unit'].to_dict()
        # 15 minutes, 30 minutes, 1 hour
        df['lag1'] = (df.index - pd.Timedelta('15 minutes')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('30 minutes')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('1 day')).map(target_map)
        df['lag4'] = (df.index - pd.Timedelta('7 days')).map(target_map)
        df['lag5'] = (df.index - pd.Timedelta('15 days')).map(target_map)
        df['lag6'] = (df.index - pd.Timedelta('30 days')).map(target_map)
        df['lag7'] = (df.index - pd.Timedelta('45 days')).map(target_map)
    except KeyError as e:
        print(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    return df


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