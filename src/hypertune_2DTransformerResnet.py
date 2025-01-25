from pathlib import Path
from typing import Dict

import ray
import torch
from loguru import logger
from ray import tune
from hypertuner import Hypertuner
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from settings import base_hypertuner
from mltrainer.preprocessors import BasePreprocessor
from logs_loader import *

def hypertune_2DTransformerResnet():

    ray.init()
    
    config = {
        "preprocessor": BasePreprocessor,
        "tune_dir": base_hypertuner.tune_dir,
        "data_dir": base_hypertuner.data_dir,
       # "batch": tune.choice([8, 16, 32]),  # Batch size specific to the dataset
        "batch" : 16,
        #"hidden": tune.choice([16, 32, 64]),
        "hidden": 128,
        #"dropout": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "dropout": 0.3,
        "num_layers": tune.randint(2, 5),
        "model_type": "2DTransformerResnet",  # Specify the model type
        #"model_type": tune.choice(["2DCNN", "2DCNNResnet"]),  # Specify the model type
        'num_blocks' : tune.randint(1, 5),
        "num_blocks": 3,
        'num_classes' : 5,
        'shape' : (8, 24),
        #"num_heads": tune.choice([2, 4, 8, 16]),
        "num_heads": 8,
        #"scheduler": tune.choice([torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR]),
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "factor": tune.choice([0,1, 0.2, 0.3, 0.4]),
        #"patience": 3,
        "patience": tune.choice([2, 3]),
        
    }
    hypertuner = Hypertuner(config)
    hypertuner.NUM_SAMPLES=15
    hypertuner.MAX_EPOCHS=40
    
    config["trainfile"], config["testfile"] = hypertuner.load_datafiles()
    

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

    ray.shutdown()

if __name__ == "__main__":
    hypertune_2DTransformerResnet()