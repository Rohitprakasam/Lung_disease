import argparse
import os
from Classes.Model import LungModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lung Disease Classifier CNN")
    parser.add_argument("--data", type=str, default="Data", help="Path to data directory containing train/val/test")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    args = parser.parse_args()

    # Create & Train Model
    print(f"Initializing LungModel with data_dir: {args.data}")
    model = LungModel(data_dir=args.data)
    model.train(epoch_size=args.epochs)
