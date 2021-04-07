import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb
from rossmann_store_sales import config, pipeline, evaluation
from rossmann_store_sales import __version__ as _version

def save_pipeline(*, pipeline_to_persist, model_to_persist) -> None:
    """Persist the pipeline."""
    save_file_pipeline = f"pipeline_version_{_version}.pkl"
    save_path_pipeline = config.TRAINED_MODEL_DIR / save_file_pipeline

    save_file_model = f"model_version_{_version}.pkl"
    save_path_model = config.TRAINED_MODEL_DIR / save_file_model

    joblib.dump(pipeline_to_persist, save_path_pipeline)
    joblib.dump(model_to_persist, save_path_model)

    print(f'Pipeline version {_version} saved.')


def apply_pipeline_steps() -> None:
    """Apply all the pipeline stesps in the training data"""
    train = config.TRAIN
    X = pipeline.pipeline.fit_transform(train)
    y = np.log1p(train.loc[X.index, config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    num_round = 20000
    evallist = [(dtrain, 'train'), (dtest, 'test')]

    plst = config.HYPERPARAMETERS.items()

    # training
    print('*** TRAINING... ***')
    bst = xgb.train(plst, dtrain, num_round, evallist, feval=evaluation.rmspe_xg, verbose_eval=250, early_stopping_rounds=250)

    save_path_X = config.DATASET_DIR / f'training_data_preprocessed_v{_version}.csv'
    save_path_y = config.DATASET_DIR / f'target_v{_version}.csv'

    # Save data
    X.to_csv(save_path_X, index=False)
    y.to_csv(save_path_y, index=False)
    print('Training data preprocessed was saved.')

    # Save pipeline
    save_pipeline(pipeline_to_persist=pipeline.pipeline, model_to_persist=bst)


if __name__ == '__main__':
    apply_pipeline_steps()