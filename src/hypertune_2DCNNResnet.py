import random
import ray
import torch
from ray import tune
from hypertuner import Hypertuner
from settings import base_hypertuner, modelnames, config_param
from mltrainer.preprocessors import BasePreprocessor

def hypertune_2DCNN():
    """
    Hypertuning function to set parameters for 2D CNN model
    and start hypertuning with ray
    
    """    
    ray.init()

    config = {
        config_param.preprocessor: BasePreprocessor,
        #config_param.optimizer: tune.choice([torch.optim.Adam, torch.optim.AdamW]),
        config_param.optimizer: torch.optim.Adam,
        config_param.tune_dir: base_hypertuner.tune_dir,
        config_param.data_dir: base_hypertuner.data_dir,
        config_param.seed: random.randint(0, 2**32 - 1),
        config_param.batch: tune.choice([16, 32]),  # Batch size specific to the dataset
        config_param.hidden: tune.choice([128, 256]), # hidden units for cnn and dense layer
        config_param.dropout: tune.choice([0.2, 0.3, 0.4]),
        config_param.num_layers: tune.randint(2, 5), #num layers RNN
        config_param.model_type: modelnames.CNN2DResNet,  # Specify the model type
        config_param.num_blocks: tune.randint(1, 8), # num conv / resnet blocks
        config_param.num_classes: 5,
        config_param.shape: (16, 12), # shape for 2D models
        config_param.num_heads: tune.choice([4, 8]), # heads for attention
        config_param.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        config_param.factor:tune.choice([0.2, 0.3, 0.4]),
        config_param.patience: 2, # wait time before reducing lr with factor
        config_param.earlystopping_patience: 15, # wait time if netwerk is not learning before stopping
        config_param.input_length:192, #input for gru/cnn model,
        #config_param.traindataset: tune.choice(["smote", "oversampled"])
        config_param.traindataset: "smote"
    }

    hypertuner = Hypertuner(config)
    hypertuner.NUM_SAMPLES=15
    hypertuner.MAX_EPOCHS=30
    
    config["trainfile"], config["testfile"] = hypertuner.load_datafiles()    

    ray.shutdown()

if __name__ == "__main__":
    hypertune_2DCNN()
    