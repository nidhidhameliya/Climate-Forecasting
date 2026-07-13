import argparse
import sys

from climate_forecasting.utils.pipeline_utils import (
    load_config,
    print_run_summary,
    run_script,
)


def run_pipeline(args):
    """Orchestrates the climate forecasting pipeline based on command-line arguments."""
    config = load_config()
    print_run_summary(config)

    if args.preprocess:
        print("--- Starting Preprocessing Stage ---")
        # Note: merge_years.py is a manual, one-time step.
        run_script("preprocessing/subset_region.py")
        run_script("preprocessing/split_by_year.py")
        run_script("preprocessing/resample_time.py")
        run_script("preprocessing/normalize.py")
        run_script("preprocessing/create_sequences.py")
        print("--- Preprocessing Stage Complete ---")

    if args.train:
        print("--- Starting Training Stage ---")
        run_script("training/train.py")
        print("--- Training Stage Complete ---")

    if args.test:
        print("--- Starting Testing Stage ---")
        # The new standard is to use run_evaluation.py for all post-training analysis
        # which includes testing, metric calculation, and visualization.
        run_script("run_evaluation.py")
        print("--- Testing Stage Complete ---")


def main():
    """Parses command-line arguments and triggers the pipeline."""
    parser = argparse.ArgumentParser(description="Climate Forecasting Pipeline Orchestrator")

    parser.add_argument("--preprocess", action="store_true", help="Run the full data preprocessing pipeline.")
    parser.add_argument("--train", action="store_true", help="Run the model training pipeline.")
    parser.add_argument("--test", action="store_true", help="Run the model testing pipeline.")

    args = parser.parse_args()

    if not any([args.preprocess, args.train, args.test]):
        parser.print_help()
        print("\n⚠️ Please specify at least one stage to run: --preprocess, --train, or --test")
        sys.exit(1)

    try:
        run_pipeline(args)
        print("\n🎉 Pipeline finished successfully!")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\n🔥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()