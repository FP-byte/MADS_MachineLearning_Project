from pathlib import Path
from typing import Dict
import tomllib

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor
from mads_datasets.base import BaseDatastreamer
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from mads_datasets import DatasetFactoryProvider, DatasetType
from metrics import Accuracy, F1Score, Precision, Recall
import datasets
from settings import base_hypertuner

class Hypertuner:
    def __init__(self, settings_hypertuner: Dict, config: Dict):
        """
        Hypertuner class to handle training with Ray Tune for hyperparameter optimization.

        Args:
            settings_hypertuner (Dict): General settings for the hypertuner.
            config (Dict): Hyperparameter configuration to tune.
        """
        self.NUM_SAMPLES = settings_hypertuner.get("NUM_SAMPLES", 10)
        self.MAX_EPOCHS = settings_hypertuner.get("MAX_EPOCHS", 50)
        self.device = settings_hypertuner.get("device", "cpu")
        self.accuracy = settings_hypertuner.get("accuracy", metrics.Accuracy())
        self.f1micro = settings_hypertuner.get("f1micro", F1Score(average='micro'))
        self.f1macro = settings_hypertuner.get("f1macro", F1Score(average='macro'))
        self.precision = settings_hypertuner.get("precision", Precision('micro'))
        self.recall = settings_hypertuner.get("recall", Recall('macro'))
        self.reporttypes = settings_hypertuner.get("reporttypes", [ReportTypes.RAY, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW])
        self.config = config

        self.search = HyperOptSearch()
        self.scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration", grace_period=1, reduction_factor=3
        )
        self.reporter = CLIReporter()
        self.reporter.add_metric_column("Accuracy")

    def shorten_trial_dirname(self, trial):
        """Shorten the trial directory name to avoid path length issues."""
        return f"trial_{trial.trial_id}"

    def train(self, config):
        """
        Train function to be passed to Ray Tune. Dynamically handles datasets and models.

        Args:
            config (Dict): Hyperparameter configuration provided by Ray Tune.
        """               
                
        data_dir = config["data_dir"]
        
        trainfile = Path(config["trainfile"])
        testfile = Path(config["testfile"]) 


        # load the data based on the configuration
        if config.get("model_type") in ["1DTransformer", "1DTransformerResnet"]:
            traindataset = datasets.HeartDataset1D(trainfile, target="target")
            testdataset = datasets.HeartDataset1D(testfile, target="target")

        else:                
            traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=config["shape"])
            testdataset = datasets.HeartDataset2D(testfile, target="target", shape=config["shape"])
   

        #Load the datastreamers
        preprocessor_class = config.get("preprocessor", BasePreprocessor)
        preprocessor = preprocessor_class()
        
        with FileLock(data_dir / ".lock"):
            trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=32)
            teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=32)


        # Initialize the model
        model = self._initialize_model(config)


        # Trainer settings
        trainersettings = TrainerSettings(
            epochs=self.MAX_EPOCHS,
            metrics=[self.accuracy, self.f1micro, self.f1macro, self.precision, self.recall],
            logdir=Path("."),
            train_steps=len(trainstreamer)//5,
            valid_steps=len(teststreamer)//5,
            reporttypes=self.reporttypes,
            scheduler_kwargs=None,
            earlystop_kwargs=None,
        )
        if config.get("scheduler") == "torch.optim.lr_scheduler.OneCycleLR":
            print("Using OneCycleLR")
            trainersettings.scheduler_kwargs = {"max_lr": 0.01, "steps_per_epoch": trainersettings.train_steps}
        else:
            trainersettings.scheduler_kwargs = {"factor": config['factor'], "patience": config['patience']}

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("Using MPS")
        else:
            device = "cpu"

        # Set up the trainer
        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
            #device=device,
        )

        logger.info(f"Starting training on {self.device}")
        trainer.loop()
    
  
    def load_datafiles(self, data_dir: Path):
        configfile = Path("config.toml")

        with configfile.open('rb') as f:
            paths = tomllib.load(f)

        
        tune_dir = Path("models/ray").resolve()
        if not tune_dir.exists():
            tune_dir.mkdir(parents=True)
            logger.info(f"Created {tune_dir}")

        #load train and test files
        trainfile = data_dir / (paths['arrhythmia_oversampled'] + '_train.parq')
        testfile = data_dir / (paths['arrhythmia'] + '_test.parq')
        return trainfile, testfile


    def _initialize_model(self, config):
        """
        Initialize and return the model based on the configuration.

        Args:
            config (dict): A dictionary containing the configuration parameters. 
                           It must include the key "model_type" which specifies 
                           the type of model to initialize. Supported model types 
                           are "2DCNN", "2DCNNResnet", "1DTransormer", "2DTransformer", 
                           "1DTransformerResnet", and "2DTransformerResnet".

        Returns:
            object: An instance of the specified model class.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        """Initialize and return the model based on the configuration."""
        
        model_type = config.get("model_type", "2DCNN")

        if model_type == "2DCNN":
            from models import CNN
            return CNN(config)
        elif model_type == "2DCNNResnet":
            from models import CNN2DResNet
            return CNN2DResNet(config)
        elif model_type == "1DTransormer":
            from models import Transformer
            return Transformer(config)
        elif model_type == "2DTransformer":
            from models import Transformer2D
            return Transformer2D(config)
        elif model_type == "1DTransformerResnet":
            from models import Transformer1DResnet
            return Transformer1DResnet(config)
        elif model_type == "2DTransformerResnet":
            from models import Transformer2DResNet
            return Transformer2DResNet(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    #test with 2DCNN
    ray.init()
    
    data_dir = base_hypertuner.data_dir
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
        "tune_dir": base_hypertuner.tune_dir,
        "data_dir": data_dir,
        "batch": 32,  # Batch size specific to the dataset
        "hidden": tune.choice([64, 128, 256, 512]),
        "dropout": tune.uniform(0.0, 0.4),
        "num_layers": tune.randint(2, 5),
        "model_type": "2DCNN",  # Specify the model type
        "model_type": tune.choice(["2DCNN", "2DCNNResnet", "2DTransformerResnet"]),  # Specify the model type
        'num_blocks' : tune.randint(1, 5),
        'num_classes' : 5,
        'shape' : (16, 12),
        "num_heads": 8,
        "scheduler": tune.choice(["torch.optim.lr_scheduler.ReduceLROnPlateau", "torch.optim.lr_scheduler.OneCycleLR"]),
        "factor": tune.uniform(0.2, 0.9),
        "patience": tune.randint(2, 4),
        
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