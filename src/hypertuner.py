from pathlib import Path
from typing import Dict
import tomllib
import os
from datetime import datetime
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
from loguru import logger
import logging
import tempfile


class Hypertuner:
    def __init__(self, config: Dict):
        """
        Hypertuner class to handle training with Ray Tune for hyperparameter optimization.

        Args:
            settings_hypertuner (Dict): General settings for the hypertuner.
            config (Dict): Hyperparameter configuration to tune.
        """

        self.NUM_SAMPLES = base_hypertuner.NUM_SAMPLES
        self.MAX_EPOCHS = base_hypertuner.MAX_EPOCHS      
        self.accuracy = base_hypertuner.accuracy
        self.f1micro = base_hypertuner.f1micro
        self.f1macro = base_hypertuner.f1macro
        self.precision = base_hypertuner.precision
        self.recall = base_hypertuner.recall
        self.reporttypes = base_hypertuner.reporttypes
        self.device = base_hypertuner.device
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

    
    def generate_valid_hidden_and_heads(self, hidden: list, heads : int):
            valid_combinations = []
            heads = int(heads)  # Convert heads to int
            for h in range(hidden[0], hidden[1], 20):  # range of hidden units
                print(type(h))
                heads = int(heads)
                if h % heads == 0:
                    print(type(heads))
                    valid_combinations.append(hidden)
            print(f"Valid combinations: {valid_combinations}")
            return random.choice(valid_combinations)

    def test_best_model(self, best_result, smoke_test=False):

        best_trained_model = self._initialize_model(best_result.config["model_type"])
        
        best_trained_model.to(self.device)

        checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

        model_state, optimizer_state = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state)

        if smoke_test:
            _, testset = load_test_data()
        else:
            _, testset = load_data()

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = best_trained_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print("Best trial test set accuracy: {}".format(correct / total))


    def train(self, config):
        """
        Train function to be passed to Ray Tune. Dynamically handles datasets and models.

        Args:
            config (Dict): Hyperparameter configuration provided by Ray Tune.
        """               
                
        data_dir = config["data_dir"]
        
        trainfile = Path(config["trainfile"])
        testfile = Path(config["testfile"]) 

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print('MPS is available')
        else:
            self.device = torch.device('cpu')
    

        print(f"Training with model: {config['model_type']}")
        # load the data based on the configuration
        if config["model_type"] in ["1DTransformer", "1DTransformerResnet", "1D"]:
            print("Loading 1D data")
            traindataset = datasets.HeartDataset1D(trainfile, target="target")
            testdataset = datasets.HeartDataset1D(testfile, target="target")


        if config["model_type"] in ["2DTransformer", "2DTransformerResnet", "2DCNNResnet"]:
            print("Loading 2D data")            
            traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=config["shape"])
            testdataset = datasets.HeartDataset2D(testfile, target="target", shape=config["shape"])
   

        #Load the datastreamers
        preprocessor_class = config.get("preprocessor", BasePreprocessor)
        preprocessor = preprocessor_class()
        
        with FileLock(data_dir / ".lock"):
            trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=config["batch"])
            teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=config["batch"])


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
            #scheduler_kwargs={"factor": config['factor'], "patience": config['patience']},
            scheduler_kwargs={"factor": 0.3, "patience": 2}, #hypertuning shows default values work best
            earlystop_kwargs=None,
        )
        if config.get("scheduler") == torch.optim.lr_scheduler.ExponentialLR:
            print("Using OneCycleLR")
            trainersettings.scheduler_kwargs = {"gamma": config['factor']}
        if config.get("scheduler") == torch.optim.lr_scheduler.LambdaLR:
            # Parameters
            num_warmup_steps = 1000
            num_training_steps = 10000
            initial_lr = 5e-5
            min_lr = 1e-7

            # Custom linear scheduler with warmup
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    return max(min_lr / initial_lr, (1.0 - progress))

            # Learning rate scheduler
            trainersettings.scheduler_kwargs = {"lr_lambda": lr_lambda}
        if config.get("scheduler") == torch.optim.lr_scheduler.CosineAnnealingLR:
            trainersettings.scheduler_kwargs = {"T_max": trainersettings.train_steps}


        # Set up the trainer
        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=config.get("scheduler"),
            #scheduler=hypertuner.scheduler,
            device=self.device,
        )

        logger.info(f"Starting training on {self.device}")
        try:
            trainer.loop()
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}")
            logger.warning("Training failed, error: {e}")
            raise
  
    def load_datafiles(self):
        data_dir = self.config["data_dir"]
        configfile = Path("config.toml")

        with configfile.open('rb') as f:
            paths = tomllib.load(f)

        
        tune_dir = Path("models/ray").resolve()
        if not tune_dir.exists():
            tune_dir.mkdir(parents=True)
            logger.info(f"Created {tune_dir}")

        #load train and test files
        trainfile = data_dir / (paths['arrhythmia_oversampled'] + '_train.parq')
       # trainfile = data_dir / (paths['arrhythmia_semioversampled'] + '_train.parq')
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
        elif model_type == "2DTransformer":
            from models import Transformer2D
            return Transformer2D(config)
        elif model_type == "1DTransformer":
            from models import Transformer
            return Transformer(config)
        elif model_type == "1DTransformerResnet":
            from models import Transformer1DResnet
            return Transformer1DResnet(config)
        elif model_type == "1DTransformerResnetSE":
            from models import Transformer1DResnetSE
            return Transformer1DResnetSE(config)
        elif model_type == "1DTransformerResnetSEwithAttention":
            from models import Transformer1DResnetSEwithAttention
            return Transformer1DResnetSEwithAttention(config)       
        elif model_type == "2DTransformerResnet":
            from models import Transformer2DResNet
            return Transformer2DResNet(config)
        elif model_type == "2DTransformerResnetSE":
            from models import Transformer2DResNetSE
            return Transformer2DResNetSE(config)
        elif model_type == "2DTransformerResNetWithAttention":
            from models import Transformer2DResNetWithAttention
            return Transformer2DResNetWithAttention(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    #test with 2DCNN
    #ray.init()
    ray.init(logging_level=logging.WARNING)
    

    config = {
        "preprocessor": BasePreprocessor,
        "tune_dir": base_hypertuner.tune_dir,
        "data_dir": base_hypertuner.data_dir,
        "batch": tune.choice([32, 48, 60]),  # Batch size specific to the dataset
        "hidden": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        "num_layers": tune.randint(2, 5),
        #"model_type": "2DCNNResnet",  # Specify the model type
        "model_type": "2DTransformerResnet",  # Specify the model type
        'num_blocks' : tune.randint(1, 5),
        'num_classes' : 5,
        'shape' : (16, 12),
        "num_heads": tune.choice([1, 2, 4, 8]),
       # "scheduler": tune.choice([torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.OneCycleLR]),
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "factor": 0.4,
        "patience": 2,
        
    }

    hypertuner = Hypertuner(config)
    #test setting
    hypertuner.MAX_EPOCHS=1
    hypertuner.NUM_SAMPLES=1
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
        resume=True,  # This will resume from the last checkpoint if available
        trial_dirname_creator=hypertuner.shorten_trial_dirname,
    )


    # Print the best result
    # print("Best accuracy: ", analysis.get_best_config(metric="accuracy", mode="max"))
    # print("Best recall: ", analysis.get_best_config(metric="recall", mode="max"))
    # print("Best model config: ", analysis.get_best_result(metric="recall", mode="max").config)


    best_result = analysis.get_best_trial("accuracy", "max")
    best_result = analysis.get_best_trial("recall", "max")
    print(best_result)

    print("Best trial config: {}".format(best_result.config))
  
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    print("Best trial final validation recall: {}".format(
        best_result.get_best_config(metric="recall", mode="max")))

    #hypertuner.test_best_model(best_result, smoke_test=False)

    ray.shutdown()