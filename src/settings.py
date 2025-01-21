#from pydantic import BaseModel
from dataclasses import dataclass
from ray import tune
from mltrainer import ReportTypes, metrics
from pathlib import Path
import metrics

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

base_hypertuner = baseHypertuner(
        data_dir = Path("../data/processed").resolve(),
        tune_dir= Path("models/ray").resolve(),
        SAMPLE_INT = tune.search.sample.Integer,
        SAMPLE_FLOAT = tune.search.sample.Float,
        NUM_SAMPLES = 10,
        MAX_EPOCHS = 27,
        device = "cpu",
        accuracy = [metrics.Accuracy(), metrics.F1Score(average='micro'), metrics.F1Score(average='macro'), metrics.Precision('micro'), metrics.Recall('macro'), metrics.Accuracy()],
        reporttypes = [ReportTypes.GIN, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        )

