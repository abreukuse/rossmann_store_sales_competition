import pandas as pd
import pathlib
import rossmann_store_sales

PACKAGE_ROOT = pathlib.Path(rossmann_store_sales.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

TARGET = 'Sales'

TRAIN = pd.read_csv(DATASET_DIR / 'train.csv')
STORE = pd.read_csv(DATASET_DIR / 'store.csv')
TEST = pd.read_csv(DATASET_DIR / 'test.csv')

TRAIN_DATA = ['Store',
              'Customers', 
              'Date',
              'DayOfWeek',
              'Open',
              'Promo',
              'SchoolHoliday',
              'StateHoliday',
              'Sales']

STORE_DATA = ['Store',
              'StoreType',
              'Assortment',
              'CompetitionDistance',
              'CompetitionOpenSinceMonth',
              'CompetitionOpenSinceYear',
              'PromoInterval'] 

CONVERT_TO_DATETIME = ['Date']

DATE_COLUMN = 'Date'

SEASONAL_FEATURES = ['day',
                     'week',
                     'month',
                     'year',
                     'dayofyear']

CONVERT_TO_NUMERICAL_CATEGORIES = ['StateHoliday',
                                   'StoreType',
                                   'Assortment']

TO_DROP = ['CompetitionOpenSinceYear',
           'CompetitionOpenSinceMonth',
           'PromoInterval',
           'Date',
           'DateInt',
           'Customers',
           'Sales']                                   