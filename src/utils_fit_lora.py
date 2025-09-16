import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 1.5")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    return parser.parse_args()
