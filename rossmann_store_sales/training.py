import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from rossmann_store_sales import config, pipeline, evaluation
from rossmann_store_sales import __version__ as _version
import os, shutil

def move_older_version():
    """Move older pipeline version to another directory"""
    files = [file for file in os.listdir(config.TRAINED_MODEL_DIR) if file.endswith('.pkl')]
    print('Moving last version to another directory')
    if not os.path.isdir(config.OLDER_VERSIONS):
        os.mkdir(config.OLDER_VERSIONS)

    if len(files) > 0:
        source = config.TRAINED_MODEL_DIR
        destination = config.OLDER_VERSIONS
        for file in files:
            shutil.move(f'{source}/{file}', destination)


def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    move_older_version()

    save_file_pipeline = f"pipeline_version_{_version}.pkl"
    save_path_pipeline = config.TRAINED_MODEL_DIR / save_file_pipeline

    joblib.dump(pipeline_to_persist, save_path_pipeline)
    print(f'Pipeline version {_version} saved.')


def apply_pipeline_steps() -> None:
    """Apply all the pipeline stesps in the training data"""
    X = config.TRAIN
    y = np.log1p(X.loc[X.index, config.TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)
    
    X_train = pipeline.pipeline[:-1].fit_transform(X_train)
    X_test = pipeline.pipeline[:-1].transform(X_test)

    y_train = y_train.iloc[X_train.index]
    y_test = y_test.iloc[X_test.index]

    # training
    print('*** TRAINING... ***')
    pipeline.pipeline[-1].fit(X_train, y_train,
                              eval_metric=evaluation.rmspe_xg,
                              eval_set=[(X_train, y_train), (X_test, y_test)],
                              early_stopping_rounds=250,
                              verbose=True)

    save_path_X = config.DATASET_DIR / f'training_data_preprocessed_v{_version}.csv'
    save_path_y = config.DATASET_DIR / f'target_v{_version}.csv'

    # Save data
    X_train.to_csv(save_path_X, index=False)
    y_train.to_csv(save_path_y, index=False)
    print('Training data preprocessed was saved.')

    # Save pipeline
    save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == '__main__':
    apply_pipeline_steps()