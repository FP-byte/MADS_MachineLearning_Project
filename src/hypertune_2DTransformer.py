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

def hypertune_Transformer():

    ray.init()
    
    data_dir = base_hypertuner.data_dir
    settings_hypertuner = {       
        "NUM_SAMPLES": 15,
        "MAX_EPOCHS": 30,
        "device": base_hypertuner.device,
        "accuracy": base_hypertuner.accuracy,            
        "f1micro": base_hypertuner.f1micro,
        "f1macro": base_hypertuner.f1macro,
        "precision": base_hypertuner.precision,
        "recall" : base_hypertuner.recall,
        "reporttypes": base_hypertuner.reporttypes,
    }

    config = {
        "preprocessor": BasePreprocessor,
        "tune_dir": base_hypertuner.tune_dir,
        "data_dir": data_dir,
        "batch": tune.choice([16, 32]),  # Batch size specific to the dataset
        "hidden": tune.choice([64, 128]),
        "dropout": tune.uniform(0.2, 0.4),
        "num_layers": tune.randint(2, 5),
        #"model_type": "2DTransformer",  # Specify the model type
        "model_type": tune.choice(["2DTransformerResnet", "2DCNNResnet"]),  # Specify the model type
        'num_blocks' : tune.randint(1, 5),
        'num_classes' : 5,
        'shape' : (16, 12),
        "num_heads": tune.choice([4, 8]),
       # "scheduler": tune.choice([torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR]),
        #"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau, # using tuner scheduler
        "factor": tune.choice([0.2, 0.3, 0.4]),
        "patience": 3,
        
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

    ray.shutdown()

if __name__ == "__main__":
    hypertune_Transformer()