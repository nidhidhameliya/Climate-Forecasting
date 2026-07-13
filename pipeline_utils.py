import os
import subprocess
import sys
import yaml

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_script(script_path):
    """Runs a Python script as a module from the project root."""
    print(f"\n🚀 Running {script_path} ...")

    # Convert file path to module path (e.g., 'preprocessing/train.py' -> 'preprocessing.train')
    module_path = script_path.replace("/", ".").replace("\\", ".").replace(".py", "")

    result = subprocess.run(
        [sys.executable, "-m", module_path],
        cwd=PROJECT_ROOT,
        check=False  # Don't raise CalledProcessError automatically
    )

    if result.returncode != 0:
        raise RuntimeError(f"❌ Error while running {script_path}. Return code: {result.returncode}")

    print(f"✅ Finished {script_path}")


def load_config():
    """Loads the main configuration file."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError("❌ config.yaml not found in project root.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_run_summary(config):
    """Prints a summary of the pipeline configuration."""
    print("\n==============================")
    print("📊 Climate Forecasting Pipeline")
    print("==============================")
    print(f"Variable: {config['variable']}")
    print(f"Model: {config['model']['name']}")
    print(f"Sequence Length: {config['sequence_length']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Region: {config.get('region', 'N/A')}")
    print("==============================\n")