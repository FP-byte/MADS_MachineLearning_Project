#from pydantic import BaseModel
from dataclasses import dataclass
from ray import tune
from mltrainer import ReportTypes, metrics
from pathlib import Path
from metrics import Accuracy, F1Score, Precision, Recall

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
        SAMPLE_INT = tune.search.sample.Integer,
        SAMPLE_FLOAT = tune.search.sample.Float,
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

