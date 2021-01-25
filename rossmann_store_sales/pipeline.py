from rossmann_sales import config, functions
from sklearn.pipeline import Pipeline
from sklearn.compose import FunctionTransformer
from sklearn.impute import SimpleImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engineering_time_series import seasonal_features

pipeline = Pipeline([
    (
        'select',
        FunctionTransformer(functions.select, kw_args={'variables': config.TRAIN_DATA})
    ),
    
    (
        'merge',
        FunctionTransformer(functions.merge, kw_args={'variables': config.STORE_DATA})
    ),

    (
        'convert_to_datetime',
        FunctionTransformer(functions.convert_to_datetime, kw_args={'variables': config.CONVERT_TO_DATETIME})
    ),

    (
        'remove_store_open_no_sales',
        FunctionTransformer(functions.remove_store_open_no_sales, kw_args={'target': config.TARGET})
    ),

    (
        'seasonal_features',
        FunctionTransformer(functions.seasonal_features, kw_args={'date_column': config.DATE_COLUMN,
                                                                  'which_ones': config.SEASONAL_FEATURES,
                                                                  'deliver': True})
    ),

    (
        'convert_dates_toInt',
        FunctionTransformer(functions.convert_dates_toInt)
    ),

    (
        'convert_to_numerical_categories',
        FunctionTransformer(functions.convert_to_numerical_categories, kw_args={'variables': config.CONVERT_TO_NUMERICAL_CATEGORIES})
    ),

    (
        'convert_CompetitionSince_to_int',
        FunctionTransformer(functions.convert_CompetitionSince_to_int)
    ),

    (
        'PromoInterval_processing',
        FunctionTransformer(functions.PromoInterval_processing)
    ),

    (
        'to_drop',
        FunctionTransformer(functions.to_drop, kw_args={'variables': config.TO_DROP})
    ),

    (
        'SimpleImputer',
        SklearnTransformerWrapper(transformer=SimpleImputer(strategy='constant', fill_value=-1))
    )
])