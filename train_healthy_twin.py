import argparse
from Classes.HealthyTwinTrainer import HealthyTwinTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and calibrate Healthy Twin autoencoder")
    parser.add_argument("--data", type=str, default="Data", help="Path to Data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--out", type=str, default="runs", help="Output directory")
    args = parser.parse_args()

    trainer = HealthyTwinTrainer(data_dir=args.data, out_dir=args.out)
    trainer.train_and_calibrate(epochs=args.epochs)

