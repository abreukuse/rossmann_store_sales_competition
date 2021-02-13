import pandas as pd
import numpy as np
import joblib
from rossmann_store_sales import config, pipeline

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""
    save_file_name = 'pipeline_010.pkl'
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print('Pipeline saved.')

def apply_pipeline_steps() -> None:
    """Apply all the pipeline stesps in the training data"""
    train = config.TRAIN
    train_preprocessed = pipeline.pipeline.fit_transform(train)
    save_path = config.DATASET_DIR / 'train_preprocessed.csv'
    train_preprocessed.to_csv(save_path, index=False)
    print('Training data preprocessed was saved.')

    save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == '__main__':
    apply_pipeline_steps()