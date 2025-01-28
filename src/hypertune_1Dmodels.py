from pathlib import Path
from typing import Dict
import ray
import torch
from loguru import logger
from ray import tune
from hypertuner import Hypertuner
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from settings import base_hypertuner, modelnames
from mltrainer.preprocessors import BasePreprocessor

def hypertune_1Dmodels():

    ray.init()

    data_dir = base_hypertuner.data_dir
    
    config = {
        "preprocessor": BasePreprocessor,
        "optimizer": tune.choice([torch.optim.Adam, torch.optim.AdamW]),
        "tune_dir": base_hypertuner.tune_dir,
        "data_dir": data_dir,
        "batch": tune.choice([16, 32, 48, 60]),  # Batch size specific to the dataset
        "input": 1,
        "hidden": tune.choice([32, 64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        "num_layers": tune.randint(2, 5),
        #"model_type": "1DTransformerResnetSE",  # Specify the model type
        "model_type": tune.choice([modelnames.CNN1DResNet , modelnames.Transformer1DResnet, modelnames.Transformer1DResnetSE]),  # Specify the model type
        'num_blocks' : tune.randint(1, 8),
        'num_classes' : 5,
        'shape' : (16, 12),
        "num_heads": tune.choice([1, 2, 4, 8]),
       # "scheduler": tune.choice([torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR]),
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "factor": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "patience": 2,
        'earlystopping_patience': 8,
        "input_length":192
        
    }


    hypertuner = Hypertuner(config)
    hypertuner.NUM_SAMPLES=15
    hypertuner.MAX_EPOCHS=20
    
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
    hypertune_1Dmodels()