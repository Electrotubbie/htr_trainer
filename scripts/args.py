import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script arguments"
    )

    parser.add_argument(
        "--slice-dataset-dir",
        type=str,
        default="dataset_slice",
        help="Path to sliced dataset directory"
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="htr_experiments",
        help="Directory to store experiment outputs"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Training batch size"
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Evaluation batch size"
    )

    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights"
    )

    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze decoder weights"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (e.g. 5e-6)"
    )

    return parser.parse_args()