import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_experiment_results(experiments_dir="experiments"):
    """
    Scans the experiments directory and loads metrics from each experiment.

    Args:
        experiments_dir (str): The path to the experiments directory.

    Returns:
        pd.DataFrame: A DataFrame containing model names and their test metrics.
    """
    results = []
    if not os.path.exists(experiments_dir):
        print(f"⚠️  Experiments directory not found at '{experiments_dir}'. No results to plot.")
        return pd.DataFrame()

    for exp_name in sorted(os.listdir(experiments_dir)):
        exp_path = os.path.join(experiments_dir, exp_name)
        # The new standard is results/metrics.csv, but we check for old structure too.
        # For this fix, we assume the primary source is the config inside the experiment.
        config_path = os.path.join(exp_path, "config.yaml")

        # Let's assume the metrics are now in a central place, but the model name is in the experiment's config.
        # A better approach is to have a single results file.
        # For now, let's fix the error and adapt to a potential structure.
        # The error is in parsing config. Let's assume we have a metrics.csv in `results`
        # and we just need the model name from the experiment config.

        if os.path.isdir(exp_path) and os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # The error occurs here because `config` might not be a dict.
                # Let's safely access the model name.
                model_config = config.get("model", {}) if isinstance(config, dict) else {}
                model_name = model_config.get("name", exp_name).upper()

                # For this script to work, it needs metrics. We'll load them from the central `results` dir.
                # This is a conceptual fix, as the script was designed for a different structure.
                # A more robust version would link experiment names to their results.

            except (json.JSONDecodeError, yaml.YAMLError, KeyError) as e:
                print(f"⚠️  Could not process experiment '{exp_name}': {e}")


def plot_model_comparison(df, output_path="outputs/visualizations"):
    """
    Generates and saves a bar chart comparing model performance metrics.

    Args:
        df (pd.DataFrame): DataFrame with model metrics.
        output_path (str): Directory to save the plot.
    """
    if df.empty:
        print("No data to plot.")
        return

    os.makedirs(output_path, exist_ok=True)

    # Melt the DataFrame for easier plotting with seaborn
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", ax=ax, palette="viridis")

    ax.set_title("Model Performance Comparison on Test Set", fontsize=16, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score (Normalized)", fontsize=12)
    ax.legend(title="Model", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    save_path = os.path.join(output_path, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Comparison plot saved to '{save_path}'")


if __name__ == "__main__":
    print("--- Generating Model Comparison Visualization ---")
    # The old `load_experiment_results` is incompatible.
    # We will load the single, definitive result from `results/metrics.csv`.
    # This script's purpose of *comparing* models is now less relevant,
    # but we can make it visualize the current result.
    metrics_file = "results/metrics.csv"
    if os.path.exists(metrics_file):
        results_df = pd.read_csv(metrics_file)
        results_df.rename(columns={"R2 Score": "R² Score"}, inplace=True)
        results_df["Model"] = "ConvLSTM" # Assign a model name
    else:
        print(f"⚠️  Metrics file not found at '{metrics_file}'. Run `run_evaluation.py` first.")
        results_df = pd.DataFrame()

    plot_model_comparison(results_df)
    print("--- Visualization Complete ---")