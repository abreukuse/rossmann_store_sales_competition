import pandas as pd
import numpy as np
from rossmann_sales import config
from sklearn.base import BaseEstimator, TransformerMixin

TRAIN = pd.read_csv(config.DATASET_DIR / config.TRAIN)

def select(X, variables):
    """Select the necessary variable for the preprocessing steps"""
    select = [variable for variable in variables if variable in X.columns]
    X = X[select].copy()
    print(f'* this variables: {select} were selected')
    return X


def merge(X, variables):
    """Join train and Store datasets"""
    X = pd.merge(X, store[variables], how='left', on=['Store'])
    print('* train data merged with store dataset')
    return X 

def convert_to_datetime(X, variables):
    """Convert 'Date' from string do datetime"""
    for variable in variables:
        X[variable] = pd.to_datetime(X[variable], format='%Y-%m-%d')
    print(f'* {variables} converted to datetime')
    return X   


def remove_store_open_no_sales(X):
    """Remove samples regeardig Store open and zero sales"""
    if config.TARGET in X.columns:
        X = X.loc[~((X['Open'] == 1) & (TRAIN[config.TARGET] == 0))]
        print(f'* {target} is in dataset columns. Some samples were removed')
        return X
    else:
        print(f'* {target} is NOT in dataset columns. None sample was removed')
        return X  


def remove_closed_days(X):
    """Only open days will ne considered"""
    X = X.loc[X['Open'] == 1, :]
    return X


def convert_dates_to_Int(X):
    """Convert 'Date' from datetime to integer"""
    X['DateInt'] = X.loc[X['Date'].notna(), ['Date']].astype(np.int64)
    print('* Date converted to int')
    return X  


def remove_long_ago_stores_data(X):
    """Remove sales samples that are too different from the rest of the data and from too long time ago"""
    indices = []
    if config.TARGET in X.columns: # execute in training data only
        store_dates_to_remove = {105:1.368e18, 163:1.368e18,
                                172:1.366e18, 364:1.37e18,
                                378:1.39e18, 523:1.39e18,
                                589:1.37e18, 663:1.39e18,
                                676:1.366e18, 681:1.37e18,
                                700:1.373e18, 708:1.368e18,
                                709:1.423e18, 730:1.39e18,
                                764:1.368e18, 837:1.396e18,
                                845:1.368e18, 861:1.368e18,
                                882:1.368e18, 969:1.366e18,
                                986:1.368e18, 192:1.421e18,
                                263:1.421e18, 500:1.421e18,
                                797:1.421e18, 815:1.421e18,
                                825:1.421e18}

        for key, value in store_dates_to_remove.items():
            index = X.query(f'(Store == {key}) & (DateInt < {value})').index.to_list()
            indices = indices + index

        X = X.drop(indices)
        print(f'* Removed long ago data')
        return X
    else:
        return X


def convert_to_numerical_categories(X, variables):
    """Convert categorical data to numerical integers"""
    for variable in variables:
        X[variable] = X[variable].astype('category').cat.codes
    print(f'* {variables} converted to numerical categories')
    return X            


def convert_CompetitionSince_to_int(X):
    """Create datetime variable from 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth' and convert to integer"""
    year = X['CompetitionOpenSinceYear'].apply(lambda a: str(int(a)) if not pd.isna(a) else np.nan)
    month = X['CompetitionOpenSinceMonth'].apply(lambda a: str(int(a)) if not pd.isna(a) else np.nan)
    date = year + '-' + month
    X['CompetitionOpenInt'] = pd.to_datetime(date[date.notna()]).astype(np.int64)
    print('* Created "CompetitionOpenInt" variable')
    return X   


def PromoInterval_processing(X):
    """Split months from 'PromoInterval' in different columns and map them into integers"""
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
    print('* "PromoInterval" dealt with')
    return X     


def median_based_outlier(X, thresh=3):
    """Detect and remove outliers in Sales"""
    if config.TARGET in X.columns:
        indices_to_remove = []
        for store in X['Store'].unique():
            points = X.loc[X['Store'] == store, config.TARGET]
            median = np.median(points)
            modified_z_score = 0.6745 * np.abs(points - median) /  np.median(np.abs(points - median))
            boolean = modified_z_score > thresh
            indices_to_remove = indices_to_remove + boolean[boolean].index.to_list()

        X = X.drop(indices_to_remove)
        print('* Removed outliers')
        return X
    else:
        return X


class SalesBasedFeatures(BaseEstimator, TransformerMixin):
    def __ini__(self):
        self.target_features = None
    
    def fit(self, X, y=None):
        store_sales = X.groupby('Store')[config.TARGET].sum()
        store_customers = X.groupby('Store')['Customers'].sum()
        store_open = X.groupby('Store')['Open'].count()

        sales_per_day = store_sales / store_open
        customers_per_day = store_customers / store_open
        sales_per_customer_per_day = sales_per_day / customers_per_day

        series = [sales_per_day, customers_per_day, sales_per_customer_per_day]

        self.target_features = pd.concat(series, axis=1).rename({0:'sales_per_day',
                                                                 1:'customers_per_day',
                                                                 2:'sales_per_customer_per_day'}, axis=1)
        print('* Created Sales based features')
        return self

    def transform(self, X):
        X = X.merge(self.target_features, how='left', left_on='Store', right_index=True)
        return X


def to_drop(X, variables):
    """Drop unnecessary variables for training"""
    drop = [variable for variable in variables if variable in X.columns]
    X = X.drop(columns=drop)
    print(f'* dropping these variables: {drop}')
    print('* Imputing missing data')
    return X  