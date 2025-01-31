from pathlib import Path
from loguru import logger
from ray.tune import ExperimentAnalysis
import ray
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from hypertuner import Hypertuner


class Evaluator(Hypertuner):
    def __init__(self):
        self.config = hypertuner.config
        self.y_pred = []
        self.y_true = []
        self.model = None
        self.teststreamer = None
    def __call__(self):
       top_config = self.load_tunelogs_data()
       self.model = self._initialize_model(top_config['model_type'])
       _, self.teststreamer = self.load_datafiles(top_config['data_dir'], top_config['testfile'])
       self.evaluate_model(top_config, self.model, self.teststreamer)
       

    def evaluate_model(self):
        testdata = self.teststreamer.stream()
        for _ in range(len(self.teststreamer)):
            X, y = next(testdata)
            
            yhat = self.model(X)
            yhat = yhat.argmax(dim=1) # we get the one with the highest probability
            y_pred.append(yhat.cpu().tolist())
            y_true.append(y.cpu().tolist())

        yhat = [x for y in y_pred for x in y]
        y = [x for y in y_true for x in y]

        cfm = confusion_matrix(y, yhat)
        cfm = cfm / np.sum(cfm, axis=1, keepdims=True)
        print(config)
        print(f'test_results={np.round(cfm[cfm > 0.3], 3)}')
        plot = sns.heatmap(cfm, annot=cfm, fmt=".3f")
        plot.set(xlabel="Predicted", ylabel="Target")
        plt.show()
        plt.savefig(f"confusion_matrix_{config['model_type']}.png")


    def load_tunelogs_data(path="models/ray") -> pd.DataFrame:
        """
        Loads the Ray Tune results from a specified directory and returns them as a DataFrame.

        Args:
            path (str): Directory path containing Ray Tune experiment logs.

        Returns:
            pd.DataFrame: Combined and cleaned results DataFrame.
        """
        tune_dir = Path(path).resolve()
        logger.info(f"Tune directory: {tune_dir}")
        if not tune_dir.exists():
            logger.warning("Model data directory does not exist. Check your tune directory path.")
            return pd.DataFrame()

        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Collect all directories within the tune_dir
        tunelogs = sorted([d for d in tune_dir.iterdir() if d.is_dir()])
        results = []

        for logs in tunelogs:
            try:
                # Load experiment analysis
                analysis = ExperimentAnalysis(logs)

                # Convert results to DataFrame
                df = analysis.dataframe()
                df.columns = [col.lower().replace("config/", "") for col in df.columns]
                df.sort_values("accuracy", inplace=True, ascending=False)

                # Add experiment name as a column
                df["experiment"] = logs.name.replace("train_", "")

                # Optionally get best trial (for debugging/logging purposes)
                best_trial = analysis.get_best_trial(metric="test_loss", mode="min")
                if best_trial:
                    logger.info(f"Best trial for {logs.name}: {best_trial}")

                # Accumulate DataFrame
                results.append(df)

            except Exception as e:
                logger.error(f"Failed to process {logs}: {e}")

        # Combine all results into a single DataFrame
        results_df = pd.concat(results, ignore_index=True)

        # Get the top 10 rows based on accuracy
        if "recallmacro" in results_df.columns:
            top_10_df = results_df.nlargest(10, "recallmacro")
            top_10_df = top_10_df[["experiment", "trial_id", "accuracy", "model_type", "test_loss", "batch", "dropout", "hidden", "num_layers", "num_heads", "recallmacro", "iterations", "factor"]]
            top_10_df.reset_index(drop=True, inplace=True)
            print(top_10_df)
            # Save the top 10 results to a CSV file
            top_10_df.to_csv("top10_results.csv", index=False)
            top10_df.reset_index(drop=True, inplace=True)
            top_config = top10_df.iloc[0].to_dict()
            print(f"Top model configurations:{top_config}")
            
        return top_config

def evaluate_model(config, model, teststreamer):

    testdata = teststreamer.stream()
    for _ in range(len(teststreamer)):
        X, y = next(testdata)
        
        yhat = model(X)
        yhat = yhat.argmax(dim=1) # we get the one with the highest probability
        y_pred.append(yhat.cpu().tolist())
        y_true.append(y.cpu().tolist())

    yhat = [x for y in y_pred for x in y]
    y = [x for y in y_true for x in y]

    cfm = confusion_matrix(y, yhat)
    cfm = cfm / np.sum(cfm, axis=1, keepdims=True)
    print(config)
    print(f'test_results={np.round(cfm[cfm > 0.3], 3)}')
    plot = sns.heatmap(cfm, annot=cfm, fmt=".3f")
    plot.set(xlabel="Predicted", ylabel="Target")
    plt.show()
    plt.savefig(f"confusion_matrix_{config['model_type']}.png")

if __name__ == "__main__":
   
    def load_tunelogs_data(path="models/ray") -> pd.DataFrame:
        """
        Loads the Ray Tune results from a specified directory and returns them as a DataFrame.

        Args:
            path (str): Directory path containing Ray Tune experiment logs.

        Returns:
            pd.DataFrame: Combined and cleaned results DataFrame.
        """
        tune_dir = Path(path).resolve()
        logger.info(f"Tune directory: {tune_dir}")
        if not tune_dir.exists():
            logger.warning("Model data directory does not exist. Check your tune directory path.")
            return pd.DataFrame()

        # Initialize Ray
        ray.init(ignore_reinit_error=True)

        # Collect all directories within the tune_dir
        tunelogs = sorted([d for d in tune_dir.iterdir() if d.is_dir()])
        results = []

        for logs in tunelogs:
            try:
                # Load experiment analysis
                analysis = ExperimentAnalysis(logs)

                # Convert results to DataFrame
                df = analysis.dataframe()
                df.columns = [col.lower().replace("config/", "") for col in df.columns]
                df.sort_values("accuracy", inplace=True, ascending=False)

                # Add experiment name as a column
                df["experiment"] = logs.name.replace("train_", "")

                # Optionally get best trial (for debugging/logging purposes)
                best_trial = analysis.get_best_trial(metric="test_loss", mode="min")
                if best_trial:
                    logger.info(f"Best trial for {logs.name}: {best_trial}")

                # Accumulate DataFrame
                results.append(df)

            except Exception as e:
                logger.error(f"Failed to process {logs}: {e}")

        # Combine all results into a single DataFrame
        results_df = pd.concat(results, ignore_index=True)

        # Get the top 10 rows based on accuracy
        if "recallmacro" in results_df.columns:
            print(results_df.columns)
            top_10_df = results_df.nlargest(20, "accuracy")
            top_10_df = top_10_df[["experiment", "trial_id", "accuracy", "model_type", "test_loss", "batch", 'optimizer', 'num_blocks', "dropout", "hidden", "num_layers", "num_heads", "recallmacro", "iterations", "factor"]]
            top_10_df.reset_index(drop=True, inplace=True)
            print(top_10_df)
            # Save the top 10 results to a CSV file
            top_10_df.to_csv("top10_results.csv", index=False)
            top_10_df.reset_index(drop=True, inplace=True)
            top_config = top_10_df.iloc[0].to_dict()
            print(f"Top model configurations:{top_config}")
            
        return top_config

top_config =load_tunelogs_data()
print(top_config)
