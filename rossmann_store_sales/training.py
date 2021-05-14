import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from rossmann_store_sales import config, pipeline, evaluation
from rossmann_store_sales import __version__ as _version
import os, shutil

def move_older_version():
    """Move older pipeline version to another directory"""
    pkl_file = [file for file in os.listdir(config.TRAINED_MODEL_DIR) if file.endswith('.pkl')]

    if not os.path.isdir(config.OLDER_VERSIONS):
        os.mkdir(config.OLDER_VERSIONS)

    if len(pkl_file) == 1:
        file = pkl_file[0]
        source = config.TRAINED_MODEL_DIR
        destination = config.OLDER_VERSIONS
        shutil.move(f'{source}/{file}', destination)


def save_pipeline(*, pipeline_to_persist, algorithm_to_persist) -> None:
    """Persist the pipeline."""

    move_older_version()

    save_file_pipeline = f"pipeline_version_{_version}.pkl"
    save_path_pipeline = config.TRAINED_MODEL_DIR / save_file_pipeline

    save_file_model = f"model_version_{_version}.pkl"
    save_path_model = config.TRAINED_MODEL_DIR / save_file_model

    joblib.dump(pipeline_to_persist, save_path_pipeline)
    joblib.dump(algorithm_to_persist, save_path_model)

    print(f'Pipeline version {_version} saved.')


def apply_pipeline_steps() -> None:
    """Apply all the pipeline stesps in the training data"""
    X = config.TRAIN
    y = np.log1p(X.loc[X.index, config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)
    
    X_train = pipeline.pipeline.fit_transform(X_train)
    X_test = pipeline.pipeline.transform(X_test)

    y_train = y_train.iloc[X_train.index]
    y_test = y_test.iloc[X_test.index]

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    num_round = 20000
    evallist = [(dtrain, 'train'), (dtest, 'test')]

    hyperparameters = config.HYPERPARAMETERS.items()

    # training
    print('*** TRAINING... ***')
    algorithm = xgb.train(params=hyperparameters, 
                          dtrain=dtrain, 
                          num_boost_round=num_round, 
                          evals=evallist, 
                          feval=evaluation.rmspe_xg, 
                          verbose_eval=250, 
                          early_stopping_rounds=250)

    save_path_X = config.DATASET_DIR / f'training_data_preprocessed_v{_version}.csv'
    save_path_y = config.DATASET_DIR / f'target_v{_version}.csv'

    # Save data
    X.to_csv(save_path_X, index=False)
    y.to_csv(save_path_y, index=False)
    print('Training data preprocessed was saved.')

    # Save pipeline
    save_pipeline(pipeline_to_persist=pipeline.pipeline, algorithm_to_persist=algorithm)


if __name__ == '__main__':
    apply_pipeline_steps()