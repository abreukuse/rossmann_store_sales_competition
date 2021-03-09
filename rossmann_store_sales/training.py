import pandas as pd
import numpy as np
import joblib
from rossmann_store_sales import config, pipeline
from rossmann_store_sales import __version__ as _version

def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""
    save_file_name = f"pipeline_version_{_version.replace('.','-')}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(file_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    print(f'Pipeline version {_version} saved.')


def remove_old_pipelines(*, file_to_keep):
    """Delete old model pipelines from the package"""
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [file_to_keep, '__init__.py', '.gitkeep']:
            model_file.unlink()


def apply_pipeline_steps() -> None:
    """Apply all the pipeline stesps in the training data"""
    train = config.TRAIN
    train_preprocessed = pipeline.pipeline.fit_transform(train)
    save_path = config.DATASET_DIR / f'train_preprocessed_{_version}.csv'
    # Save just a sample of the prepreprossed data
    train_preprocessed.head(10).to_csv(save_path, index=False)
    print('Training data preprocessed was saved.')

    save_pipeline(pipeline_to_persist=pipeline.pipeline)


if __name__ == '__main__':
    apply_pipeline_steps()