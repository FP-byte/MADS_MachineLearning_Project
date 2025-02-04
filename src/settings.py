#from pydantic import BaseModel
from dataclasses import dataclass
from ray import tune
from mltrainer import ReportTypes, metrics
from pathlib import Path
from metrics import F1Score, Precision, Recall

@dataclass
class baseHypertuner:
    """
    A class used to represent the base configuration for hyperparameter tuning.

    Attributes
    ----------
        An integer sample parameter.
        A floating-point sample parameter.
    NUM_SAMPLES : int
        The number of samples to be used in tuning.
    MAX_EPOCHS : int
        The maximum number of epochs for training.
        The device to be used for computation (e.g., 'cpu', 'cuda').
    accuracy : str
        The accuracy metric to be used.
    reporttypes : list
        A list of report types to be generated.
    data_dir : str
        The directory where the data is stored.
    tune_dir : str
        The directory where tuning results are stored.
    f1micro : str
        The F1 score (micro) metric to be used.
    f1macro : str
        The F1 score (macro) metric to be used.
    precision : str
        The precision metric to be used.
    recall : str
        The recall metric to be used.
    """

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
    """
    A class to hold the names of various machine learning models as string attributes.

    Attributes:
        CNN2D (str): Name of the 2D Convolutional Neural Network model.
        CNN2DResNet (str): Name of the 2D Convolutional Neural Network with ResNet model.
        Transformer2D (str): Name of the 2D Transformer model.
        Transformer1D (str): Name of the 1D Transformer model.
        GRU (str): Name of the Gated Recurrent Unit model.
        AttentionGRU (str): Name of the Attention-based Gated Recurrent Unit model.
        Transformer1DResnet (str): Name of the 1D Transformer with ResNet model.
        Transformer1DResnetSE (str): Name of the 1D Transformer with ResNet and Squeeze-and-Excitation model.
        Transformer1DResnetSEwithAttention (str): Name of the 1D Transformer with ResNet, Squeeze-and-Excitation, and Attention model.
        Transformer2DResNet (str): Name of the 2D Transformer with ResNet model.
        Transformer2DResnetSE (str): Name of the 2D Transformer with ResNet and Squeeze-and-Excitation model.
        Transformer2DResNetWithAttention (str): Name of the 2D Transformer with ResNet and Attention model.
        CNN1DResNet (str): Name of the 1D Convolutional Neural Network with ResNet model.
        CNN1DGRUResNet (str): Name of the 1D Convolutional Neural Network with GRU and ResNet model.
        CNN1DGRUResNetMH (str): Name of the 1D Convolutional Neural Network with GRU, ResNet, and Multi-Head Attention model.
    """
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
    """
    A class used to replace the strings in configuration parameters for the model.

    Attributes
    ----------
    preprocessor : str
        String for the preprocessor to be used.
    tune_dir : str
        String for the directory for tuning results.
    data_dir : str
        String for the directory where data is stored.
    batch : str
        String for the batch size for training.
    gru_hidden : str
        String for the number of hidden units in the GRU layer.
    hidden : str
        String for the number of hidden units in the model.
    dropout : str
        String for the dropout rate for regularization.
    num_layers : str
        String for the number of layers in the model.
    model_type : str
        String for the type of model to be used.
    num_blocks : str
        The number of blocks in the model.
    num_classes : str
        String for the number of output classes.
    shape : str
        String for the shape of the input data.
    num_heads : str
        String for the number of attention heads.
    scheduler : str
        String for the learning rate scheduler to be used.
    factor : str
        String for the factor by which the learning rate will be reduced.
    patience : str
        String for the number of epochs with no improvement after which learning rate will be reduced.
    trainfile : str
        String for the file containing training data.
    testfile : str
        String for the file containing test data.
    earlystopping_patience : str
        String for the number of epochs with no improvement after which training will be stopped.
    optimizer : str
        String for the optimizer to be used for training.
    input_length : str
        The length of the input sequences.
    input_gru : str
        String for the input size for the GRU layer.
    seed : str
        String for the random seed for reproducibility.
    traindataset : str
        String for the dataset to be used for training.
    """
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

