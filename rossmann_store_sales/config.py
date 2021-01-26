import pathlib
import rossmann_sales

PACKAGE_ROOT = pathlib.Path(rossmann_sales.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TARGET = 'Sales'

TRAIN = 'train.csv'
TEST = 'test.csv'
STORE = 'store.csv'

TRAIN_DATA = ['Store', 
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
           'Sales']                                   