#from pydantic import BaseModel
from dataclasses import dataclass
from ray import tune
from mltrainer import ReportTypes, metrics
from pathlib import Path
from metrics import Accuracy, F1Score, Precision, Recall
import time
import random
import numpy as np

@dataclass
class baseHypertuner:

    SAMPLE_INT : int
    SAMPLE_FLOAT : float
    NUM_SAMPLES: int
    MAX_EPOCHS: int
    device : str
    accuracy: str
    reporttypes: list
    data_dir: str
    tune_dir: str
    f1micro: str
    f1macro: str
    precision: str
    recall: str


base_hypertuner = baseHypertuner(
        data_dir = Path("../data/processed").resolve(),
        tune_dir= Path("models/ray").resolve(),
        SAMPLE_INT = tune.randint(1, 100),
        SAMPLE_FLOAT = tune.uniform(0.0, 1.0),
        NUM_SAMPLES = 10,
        MAX_EPOCHS = 15,
        device = "cpu",
        accuracy = metrics.Accuracy(),
        f1micro = F1Score(average='micro'),
        f1macro = F1Score(average='macro'),
        precision = Precision('macro'),
        recall = Recall('macro'),
        reporttypes = [ReportTypes.RAY, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        )

@dataclass
class ModelNames:
    CNN2D: str
    CNN2DResNet: str
    Transformer2D: str
    Transformer1D: str
    GRU: str
    AttentionGRU: str
    Transformer1DResnet: str
    Transformer1DResnetSE: str
    Transformer1DResnetSEwithAttention: str
    Transformer2DResNet: str
    Transformer2DResnetSE: str
    Transformer2DResNetWithAttention: str
    CNN1DResNet:str
    CNN1DGRUResNet:str
    CNN1DGRUResNetMH:str



modelnames = ModelNames(
    CNN2D="CNN",
    CNN2DResNet="CNN2DResNet",
    Transformer2D="Transformer2D",
    Transformer1D="Transformer",
    GRU="GRU",
    AttentionGRU="AttentionGRU",
    Transformer1DResnet="Transformer1DResnet",
    Transformer1DResnetSE="Transformer1DResnetSE",
    Transformer1DResnetSEwithAttention="Transformer1DResnetSEwithAttention",
    Transformer2DResNet="Transformer2DResNet",
    Transformer2DResnetSE="Transformer2DResNetSE",
    Transformer2DResNetWithAttention="Transformer2DResNetWithAttention",
    CNN1DResNet = "CNN1DResNet",
    CNN1DGRUResNet = "CNN1DGRUResNet",
    CNN1DGRUResNetMH="CNN1DGRUResNetMH",

)


@dataclass
class ConfigParams:
    preprocessor: str
    tune_dir: str
    data_dir: str
    batch: str
    gru_hidden: str
    hidden: str
    dropout: str
    num_layers: str
    model_type: str
    num_blocks: str
    num_classes: str
    shape: str
    num_heads: str
    scheduler: str
    factor: str
    patience: str
    trainfile: str
    testfile: str
    earlystopping_patience: str
    optimizer: str
    input_length:str
    input_gru:str
    seed:str
    traindataset:str


config_param = ConfigParams(
    preprocessor="preprocessor",
    optimizer= "optimizer",
    tune_dir="tune_dir",
    data_dir="data_dir",
    seed = "seed",
    input_gru="input",
    batch="batch",
    hidden="hidden",
    gru_hidden="gru_hidden",
    dropout="dropout",
    num_layers="num_layers",
    model_type="model_type",
    num_blocks="num_blocks",
    num_classes="num_classes",
    shape="shape",
    num_heads="num_heads",
    scheduler="scheduler",
    factor="factor",
    patience="patience",
    trainfile="trainfile",
    testfile="testfile",
    earlystopping_patience='earlystopping_patience',
    input_length="input_length",
    traindataset="traindataset"

)

