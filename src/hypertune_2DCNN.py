from pathlib import Path
from typing import Dict

import ray
import torch
from loguru import logger
from ray import tune
from hypertuner import Hypertuner
from settings import settings_hypertuner
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import PaddedPreprocessor

def hypertune_2DCNN():

    #test with 2DCNN
    ray.init()

    data_dir = settings_hypertuner["data_dir"]
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")

    tune_dir = Path("models/ray").resolve()

    settings_hypertuner = {
        "NUM_SAMPLES": 10,
        "MAX_EPOCHS": 5,
        "device": "cpu",
        "accuracy": Accuracy(),            
        "f1micro": F1Score(average='micro'),
        "f1macro": F1Score(average='macro'),
        "precision": Precision('macro'),
        "recall" : Recall('macro'),
        "reporttypes": [ReportTypes.RAY, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
    }

    config = {
        "preprocessor": BasePreprocessor,
        "tune_dir": settings_hypertuner["tune_dir"],
        "data_dir": data_dir,
        "batch": 32,  # Batch size specific to the dataset
        "hidden": tune.randint(100, 512),
        "dropout": tune.uniform(0.0, 0.4),
        "num_layers": tune.randint(2, 5),
        "model_type": "2DCNN",  # Specify the model type
        'num_blocks' : tune.randint(1, 5),
        'num_classes' : 5,
        'shape' : (16, 12),
    }

    hypertuner = Hypertuner(settings_hypertuner, config)
    config["trainfile"], config["testfile"] = hypertuner.load_datafiles(data_dir)

    analysis = tune.run(
        hypertuner.train,
        config=config,
        metric="Accuracy",
        mode="max",
        progress_reporter=hypertuner.reporter,
        storage_path=str(config["tune_dir"]),
        num_samples=hypertuner.NUM_SAMPLES,
        search_alg=hypertuner.search,
        scheduler=hypertuner.scheduler,
        verbose=1,
        trial_dirname_creator=hypertuner.shorten_trial_dirname,
    )

if __name__ == "__main__":
    hypertune_2DCNN()