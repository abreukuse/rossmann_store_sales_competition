import pandas as pd
import numpy as np
from rossmann_sales import config

train = pd.read_csv(config.DATASET_DIR / config.TRAIN)

def select(X, variables):
    select = [variable for variable in variables if variable in X.columns]
    X = X[select].copy()
    return X


def merge(X, variables):
    X = pd.merge(X, store[variables], how='left', on=['Store'])
    return X 

def convert_to_datetime(X, variables):
    for variable in variables:
        X[variable] = pd.to_datetime(X[variable], format='%Y-%m-%d')
    return X   


def remove_store_open_no_sales(X, target):
    if target in X.columns:
        X = X.loc[~((X['Open'] == 1) & (train[target] == 0))]
        return X
    else:
        return X  


def convert_dates_toInt(X):
    X['DateInt'] = X.loc[X['Date'].notna(), ['Date']].astype(np.int64)
    return X  


def convert_to_numerical_categories(X, variables):
    for variable in variables:
        X[variable] = X[variable].astype('category').cat.codes
    return X            


def convert_CompetitionSince_to_int(X):
    year = X['CompetitionOpenSinceYear'].apply(lambda a: str(int(a)) if not pd.isna(a) else np.nan)
    month = X['CompetitionOpenSinceMonth'].apply(lambda a: str(int(a)) if not pd.isna(a) else np.nan)
    date = year + '-' + month
    X['CompetitionOpenInt'] = pd.to_datetime(date[date.notna()]).astype(np.int64)
    return X   


def PromoInterval_processing(X):
    month_mapping = {
                    'Jan' : 1,
                    'Feb' : 2,
                    'Mar' : 3,
                    'Apr' : 4,
                    'May' : 5,
                    'Jun' : 6,
                    'Jul' : 7,
                    'Aug' : 8,
                    'Sept' : 9, 
                    'Oct' : 10,
                    'Nov' : 11,
                    'Dec' : 12
                    }
    X['PromoInterval0'] = X['PromoInterval'].str.split(',', expand=True)[0].replace(month_mapping)
    return X     


def to_drop(X, variables):
    drop = [variable for variable in variables if variable in X.columns]
    X = X.drop(columns=drop)
    return X  