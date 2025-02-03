from pathlib import Path
from typing import Dict
import tomllib
import os
import time
import random
import numpy as np
from datetime import datetime
import ray
import torch
from filelock import FileLock
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
from settings import base_hypertuner, modelnames, config_param
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
        logger.info("hypertuner started")
        logger.info("hypertuner started")
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

        # Function to set the random seed for reproducibility
    def set_seed(self, seed: int):
        random.seed(seed)  # Python's random seed
        np.random.seed(seed)  # Numpy random seed
        torch.manual_seed(seed)  # PyTorch seed
        torch.cuda.manual_seed_all(seed)  # CUDA seed (for multi-GPU setups)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic algorithms
        torch.backends.cudnn.benchmark = False  # Disable non-deterministic algorithms  


    def shorten_trial_dirname(self, trial):
        """Shorten the trial directory name to avoid path length issues on Windows."""
        return f"trial_{trial.trial_id}"

    
    def generate_valid_hidden_and_heads(self, hidden: list, heads : int):
            valid_combinations = []
            heads = int(heads)  # Convert heads to int
            for h in range(hidden[0], hidden[1], 20):  # range of hidden units
                logger.debug(f"Type of h: {type(h)}")
                if h % heads == 0:
                    logger.debug(f"Type of heads: {type(heads)}")
                    valid_combinations.append(hidden)
            logger.info(f"Valid combinations: {valid_combinations}")
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
                
                # to do
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                pass


        logger.info("Best trial test set accuracy: {}".format(correct / total))


    def train(self, config: Dict):
        """
        Train function to be passed to Ray Tune. Dynamically handles datasets and models.

        Args:
            config (Dict): Hyperparameter configuration provided by Ray Tune.
        """               
                
        data_dir = config[config_param.data_dir]
        self.set_seed(config[config_param.seed])
        
        trainfile = Path(config[config_param.trainfile])
        testfile = Path(config[config_param.testfile]) 

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info('MPS is available')
        else:
            self.device = torch.device('cpu')
            logger.info('MPS is not available, using CPU')
    

        logger.info(f"Training with model: {config[config_param.model_type]}")
        
        traindataset = datasets.HeartDataset1D(trainfile, target="target")
        testdataset = datasets.HeartDataset1D(testfile, target="target")
        msg = "Loading 1D data"

        if "2D" in config[config_param.model_type]: 
            msg = "Loading 2D data"     
            traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=config[config_param.shape])
            testdataset = datasets.HeartDataset2D(testfile, target="target", shape=config[config_param.shape])
            
        logger.info(msg)
   
        #Load the datastreamers
        preprocessor_class = config.get(config_param.preprocessor, BasePreprocessor)
        preprocessor = preprocessor_class()
        
        with FileLock(data_dir / ".lock"):
            trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=config[config_param.batch])
            teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=config[config_param.batch])


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
            scheduler_kwargs={"factor": config[config_param.factor], "patience": config[config_param.patience]},
            #scheduler_kwargs={"factor": 0.3, "patience": config['patience']}, #hypertuning shows default values work best
            #earlystop_kwargs={"patience": config[config_param.earlystopping_patience], "save": True},
        )
        # Custom learning rate scheduler ExponentialLR
        if config.get(config_param.scheduler) == torch.optim.lr_scheduler.ExponentialLR:
                    logger.info("Using ExponentialLR")
                    trainersettings.scheduler_kwargs = {"gamma": config[config_param.factor]}
        
        # Custom learning rate scheduler LambdaLR
        if config.get(config_param.scheduler) == torch.optim.lr_scheduler.LambdaLR:
            logger.info("Using LambdaLR")
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
        # Custom learning rate scheduler CosineAnnealingLR    
        if config.get("scheduler") == torch.optim.lr_scheduler.CosineAnnealingLR:
            logger.info("Using OneCycleLR")
            trainersettings.scheduler_kwargs = {"T_max": trainersettings.train_steps}


        # Set up the trainer
        trainer = Trainer(
            model=model,
            settings=trainersettings,
            loss_fn=torch.nn.CrossEntropyLoss(),
            optimizer=config[config_param.optimizer],
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=config.get("scheduler"),
            device=self.device,
        )

        logger.info(f"Starting training on {self.device}")
        try:
            trainer.loop()
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}")
            raise
  
    def load_datafiles(self):
        data_dir = self.config[config_param.data_dir]
        configfile = Path("config.toml")

        with configfile.open('rb') as f:
            paths = tomllib.load(f)

        
        tune_dir = Path("models/ray").resolve()
        if not tune_dir.exists():
            tune_dir.mkdir(parents=True)
            logger.info(f"Created {tune_dir}")

        #load train and test files
        if self.config[config_param.traindataset] == 'smote':
            trainfile = data_dir / (paths['arrhythmia_smote'] + '_train.parq')
            logger.info(f"Training with SMOTE dataset")
        else:
            trainfile = data_dir / (paths['arrhythmia_oversampled'] + '_train.parq')
            logger.info(f"Training with oversampled dataset")
       # trainfile = data_dir / (paths['arrhythmia_semioversampled'] + '_train.parq')
        testfile = data_dir / (paths['arrhythmia'] + '_test.parq')

        return trainfile, testfile


    def _initialize_model(self, config: Dict):
        """
        Initialize and return the model based on the configuration.

        Args:
            config (dict): A dictionary containing the configuration parameters. 
                           It must include the key "model_type" which specifies 
                           the type of model to initialize. Supported model types 
                           are "2DCNN", "2DCNNResnet", "1DTransformer", "2DTransformer", 
                           "1DTransformerResnet", and "2DTransformerResnet".

        Returns:
            object: An instance of the specified model class.

        Raises:
            ValueError: If the specified model type is not supported.
        """
        model_type = config.get(config_param.model_type, "CNN1DResnet")
        model_classes = modelnames.__dict__
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")

        

        model_class = model_classes[model_type]
        module = __import__("models", fromlist=[model_class])
        logger.info(f"model {model_class} loaded")
       
        model = getattr(module, model_class)
        return model(config)


if __name__ == "__main__":
    #test with 2DCNN
    #ray.init()
    ray.init()




    data_dir = base_hypertuner.data_dir
    seed = random.randint(0, 2**32 - 1)
    
    config = {
        config_param.preprocessor: BasePreprocessor,
        config_param.optimizer: tune.choice([torch.optim.Adam, torch.optim.AdamW]),
        config_param.tune_dir: base_hypertuner.tune_dir,
        config_param.data_dir: base_hypertuner.data_dir,
        config_param.seed: seed,
        config_param.batch: tune.choice([16, 32]),  # Batch size specific to the dataset
        config_param.hidden: tune.choice([128, 256]), # hidden units for cnn and dense layer
        config_param.dropout: tune.choice([0.2, 0.3]),
        config_param.num_layers: tune.randint(2, 5), #num layers RNN
        config_param.model_type: modelnames.Transformer1DResnet,  # Specify the model type
        config_param.num_blocks: tune.randint(1, 8), # num conv / resnet blocks
        config_param.num_classes: 5,
        #config_param.shape: (16, 12), # shape for 2D models
        config_param.num_heads: tune.choice([2, 8]), # heads for attention
        config_param.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        config_param.factor: 0.2,
        config_param.patience: 2, # wait time before reducing lr with factor
        config_param.earlystopping_patience: 15, # wait time if netwerk is not learning before stopping
        
    }
            
    hypertuner = Hypertuner(config)
    #test setting
    
    hypertuner.MAX_EPOCHS=40
    hypertuner.NUM_SAMPLES=15
    config[config_param.trainfile], config[config_param.testfile] = hypertuner.load_datafiles()

    analysis = tune.run(
        hypertuner.train,
        config=config,
        metric="Accuracy",
        mode="max",
        progress_reporter=hypertuner.reporter,
        storage_path=str(config[config_param.tune_dir]),
        num_samples=hypertuner.NUM_SAMPLES,
        search_alg=hypertuner.search,
        scheduler=hypertuner.scheduler,
        verbose=1,
        resume=True,  # This will resume from the last checkpoint if available
        trial_dirname_creator=hypertuner.shorten_trial_dirname,
    )


    #Print the best results  
    # best result accuracy 
    best_result_acc = analysis.get_best_trial("accuracy", "max")
    # best result recall
    best_result_rec = analysis.get_best_trial("recall", "max")
    logger.info("Best accuracy: ", best_result_acc)
    logger.info("Best model config: ", analysis.get_best_result(metric="accuracy", mode="max").config)
    logger.info("Best recall: ", best_result_rec)
    logger.info("Best model config: ", analysis.get_best_result(metric="recall", mode="max").config)


    ray.shutdown()