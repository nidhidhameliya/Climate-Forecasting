import argparse
import subprocess
import yaml
import os
import sys


# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_script(script_path):
    print(f"\n🚀 Running {script_path} ...")

    result = subprocess.run(
        [sys.executable, "-m", script_path.replace("/", ".").replace(".py", "")],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(f"❌ Error while running {script_path}")

    print(f"✅ Finished {script_path}")


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError("❌ config.yaml not found in project root.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_run_summary(config):
    print("\n==============================")
    print("📊 Climate Forecasting Pipeline")
    print("==============================")
    print(f"Variable: {config['variable']}")
    print(f"Model: {config['model']['name']}")
    print(f"Hidden Dim: {config['model']['hidden_dim']}")
    print(f"Sequence Length: {config['sequence_length']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Region: {config['region']}")
    print("==============================\n")


def main(args):
    config = load_config()
    print_run_summary(config)

    if args.preprocess:
        # run_script("preprocessing/merge_years.py")  # keep this commented
        run_script("preprocessing/subset_region.py")
        # script was renamed to split_by_year; update call accordingly
        run_script("preprocessing/split_by_year.py")        # FIXED ORDER
        run_script("preprocessing/resample_time.py")
        run_script("preprocessing/normalize.py")
        run_script("preprocessing/create_sequences.py")

    if args.train:
        run_script("training/train.py")

    if args.test:
        run_script("training/test.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climate Forecasting Pipeline")

    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing pipeline")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Test model")

    args = parser.parse_args()

    if not (args.preprocess or args.train or args.test):
        print("⚠️ Please specify at least one option: --preprocess, --train, or --test")
        sys.exit(1)

    main(args)