import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from ray.tune import ExperimentAnalysis
import ray
from logs_loader import Dashboard
import plotly.express as px


warnings.simplefilter(action="ignore", category=FutureWarning)
ray.init(ignore_reinit_error=True)




def main() -> None:
    """
    Main function for running the Streamlit app.

    This function manages the user interface and renders different plots
    (Parallelplot, Scatterplot, Histogram, Boxplot) based on user input. It ensures
    the experiment data is loaded only once.
    """
    # Set the layout to wide mode
    st.set_page_config(layout="wide")
    dashboard = Dashboard()
    # Load dataset into session state
    if "tunelogs" not in st.session_state:
        st.session_state.tunelogs = dashboard.load_tunelogs_data()

    st.title("Ray Tune Logs Dashboard")

    # Ensure data is not empty
    if st.session_state.tunelogs.empty:
        st.error("The dataset is empty. Please check the data source.")
        return

    # Create two columns with different widths
    col1, _, col2 = st.columns([1, 0.1, 3])

    with col1:
        # Select plot type
        plot_type = st.radio(
            "Choose a Plot Type", 
            ["Parallelplot", "Scatterplot", "Histogram", "Boxplot"],
            key="plot_type"
        )

    if plot_type == "Parallelplot":
        with col1:
           
            # Select model
            model = st.selectbox("Select the model", st.session_state.tunelogs["model_type"].unique(), key="model_select")
            model_metrics = st.session_state.tunelogs[st.session_state.tunelogs["model_type"] == model]

            # Filter top 10 models based on a metric (e.g., accuracy)
            metric = st.selectbox("Select the metric to rank models", model_metrics.columns, key="metric_select")
            top_10_models = model_metrics.nlargest(10, metric)

            # Plot top 10 models
            st.write(f"Top 10 {model} models based on {metric}")
            model_cols = ["accuracy", "test_loss", "batch", "optimizer", "num_blocks", "dropout", "hidden", "num_layers", "num_heads", "recallmacro", "iterations", "factor", "trainfile"]

            select = st.multiselect("Select the columns to plot", model_cols, key="select_columns")
            p = top_10_models[select].reset_index().dropna()
            p.drop(columns=["index"], inplace=True)
            color = st.selectbox("Select the color (categorical preferred)", select, key="color_scatter")

        with col2:
            # Add a spacer to ensure the plot is fully visible
            st.title(f"Model configurations for {model}")
            # Parallelplot with color variation
            fig = px.parallel_coordinates(p, color=color, color_continuous_scale=px.colors.diverging.Tealrose)
            fig.update_layout(
                autosize=True,
                margin=dict(l=40, r=40, t=50, b=20),
                font=dict(size=15)
            )
            st.plotly_chart(fig, use_container_width=True)
       
    # Scatterplot
    if plot_type == "Scatterplot":
        with col1:
            x_axis = st.selectbox("Select the x-axis", st.session_state.tunelogs.columns, key="x_axis_scatter")
            y_axis = st.selectbox("Select the y-axis", st.session_state.tunelogs.columns, key="y_axis_scatter")
            color = st.selectbox("Select the color (categorical preferred)", st.session_state.tunelogs.columns, key="color_scatter")

        with col2:
            # Plot scatterplot
            fig, ax = plt.subplots()
            try:
                sns.scatterplot(data=st.session_state.tunelogs, x=x_axis, y=y_axis, hue=color, palette="tab10")
                plt.xticks(rotation=45, ha="right")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to plot scatterplot: {e}")

    # Histogram
    elif plot_type == "Histogram":
        with col1:
            hist_var = st.selectbox("Select a variable for the histogram", st.session_state.tunelogs.columns, key="histogram_var")
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        with col2:
            # Plot histogram
            fig, ax = plt.subplots()
            sns.histplot(st.session_state.tunelogs[hist_var], bins=bins, kde=True)
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    # Boxplot
    elif plot_type == "Boxplot":
        with col1:
            y_var = st.selectbox("Select variable for boxplot (Y-axis)", st.session_state.tunelogs.columns, key="boxplot_y")
            x_var = st.selectbox("Select grouping variable (X-axis)", st.session_state.tunelogs.columns, key="boxplot_x")

        with col2:
            # Plot boxplot
            fig, ax = plt.subplots()
            try:
                sns.boxplot(x=x_var, y=y_var, data=st.session_state.tunelogs)
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to plot boxplot: {e}")


if __name__ == "__main__":
    main()
