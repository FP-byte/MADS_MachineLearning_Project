from pathlib import Path
from loguru import logger
from ray.tune import ExperimentAnalysis
import ray
import pandas as pd
from typing import Dict
from tabulate import tabulate


class Dashboard():
         

    def report_results(self, results_df: pd.DataFrame, model: str) -> pd.DataFrame:
        """
        Generates a report of the top results from the given DataFrame and model.

        Args:
            results_df (pd.DataFrame): DataFrame containing the results to be reported.
            model (str): The model name for which the report is to be generated.

        Returns:
            pd.DataFrame: A cleaned and sorted DataFrame containing the top results.
        """
        report = self.report_top_results(results_df, top=10)
        df = pd.DataFrame(report[model])
        df_clean = df[['iterations', 'accuracy', 'recallmacro', 'batch', 'hidden', 'dropout', 'num_layers', 'num_blocks', 'factor', 'optimizer', 'gru_hidden', 'trainfile']]
        df_clean['trainfile'] = df_clean['trainfile'].apply(lambda x: x.name.split("_")[2])
        df_clean.sort_values(by=['accuracy', 'recallmacro', 'iterations'], inplace=True, ascending=False)
        
        report_res = {'model': model}
        for col in df_clean.columns:
            if col not in ['trainfile', 'experiment', 'accuracy', 'recallmacro', 'iterations', 'optimizer']:
                val = sorted(df_clean[col].unique().tolist())
                report_res[col] = val if len(val) == 1 else [round(v, 1) if isinstance(v, float) else v for v in val]
        
        df_report_configurations = pd.DataFrame([report_res])
        self.report_top_results_md(df_report_configurations)
        self.report_top_results_md(df_clean[:2])
        return df_clean

    def report_top_results(self, results_df: pd.DataFrame, top: int = 30) -> Dict[str, list]:
        """
        Generates a report of the top results for each model type in the given DataFrame.

        Args:
            results_df (pd.DataFrame): DataFrame containing the results to be reported.
            top (int): Number of top results to include in the report for each model type.

        Returns:
            Dict[str, list]: A dictionary where keys are model types and values are lists of top results.
        """
        report = {}
        for model in results_df.model_type.unique():
            top_results = results_df[results_df.model_type == model].nlargest(top, "accuracy")
            report[model] = top_results.to_dict(orient='records')
        return report

    def create_report_md(self, report_df: pd.DataFrame) -> str:
        """
        Converts the given DataFrame to a Markdown table.

        Args:
            report_df (pd.DataFrame): DataFrame to be converted to Markdown.

        Returns:
            str: The Markdown table as a string.
        """
        # Convert to Markdown table
        markdown_table = tabulate(report_df, headers="keys", tablefmt="pipe")
        
        # Print or save the Markdown table
        print(markdown_table)
        return markdown_table

    def report_top_results_md(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a report of the configurations in the given DataFrame in Markdown format.

        Args:
            df (pd.DataFrame): DataFrame containing the configurations to be reported.

        Returns:
            pd.DataFrame: A DataFrame containing the report in Markdown format.
        """
        report = {}
        for col in df.columns:
            report[col] = "<br>".join(map(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x), df[col]))
        
        # Convert report into a DataFrame if needed
        report_df = pd.DataFrame([report])
       
        markdown_table = self.create_report_md(report_df)
        print(markdown_table)
        return report_df



    def load_tunelogs_data(self, path="models/ray") -> pd.DataFrame:
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
            
        return results_df

    def get_top_config(self, results_df: pd.DataFrame, top: int = 20) -> Dict[str, any]:
        """
        Retrieves the top configurations based on recall from the given DataFrame.

        Args:
            results_df (pd.DataFrame): DataFrame containing the results.
            top (int): Number of top configurations to retrieve.

        Returns:
            Dict[str, any]: A dictionary containing the top configuration.
        """
        # Get the top rows based on recall
        if "recallmacro" in results_df.columns:
            print(results_df.columns)
            top_10_df = results_df.nlargest(top, "recallmacro")
            top_10_df['trainfile'] = top_10_df['trainfile'].apply(lambda x: x.name)
            top_10_df = top_10_df[["experiment", "trial_id", "accuracy", "model_type", "test_loss", "batch", 'optimizer', 'num_blocks', "dropout", "hidden", "num_layers", "num_heads", "recallmacro", "iterations", "factor", "trainfile"]]
            top_10_df.reset_index(drop=True, inplace=True)
            print(top_10_df)
            # Save the top results to a CSV file
            top_10_df.to_csv("top10_results.csv", index=False)
            top_10_df.reset_index(drop=True, inplace=True)
            top_config = top_10_df.iloc[0].to_dict()
            print(f"Top model configurations: {top_config}")
        return top_config

if __name__ == "__main__":
# test the dashboard class
   dashboard = Dashboard()
   results_df = dashboard.load_tunelogs_data()
   top_config = dashboard.get_top_config(results_df)
   print(top_config)
   model = '2DCNNResnet'
   model_report = dashboard.report_results(results_df, model)
  
   model2 = 'CNN1DGRUResNet'
   model_report = dashboard.report_results(results_df, model2)

