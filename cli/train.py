#!/usr/bin/env python3
"""
train.py
CLI script for training bioacoustic classification models.
"""
import argparse
import os
import sys
import pickle
import numpy as np
import yaml

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bioacoustica.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="BioAcoustica Training CLI")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to directory containing X.pkl and Y.pkl")
    parser.add_argument("--output_dir", type=str, help="Directory to save trained models")
    parser.add_argument("--arch", type=str, choices=["custom_cnn", "mobilenet_v2", "efficientnet_b0"], help="Model architecture")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate before final layer")
    parser.add_argument("--fine_tune", action="store_true", help="Fine-tune pre-trained backbone")
    parser.add_argument("--train_size", type=float, help="Fraction of data for training")
    parser.add_argument("--class_order", type=str, nargs="+", help="Order of classes for one-hot encoding (e.g., noise gibbon)")
    parser.add_argument("--no_augment", action="store_false", dest="augment", help="Disable data augmentation")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    # Default parameters
    params = {
        "data_dir": args.data_dir or "data/processed",
        "output_dir": args.output_dir or "models",
        "arch": args.arch or "baseline_cnn",
        "epochs": args.epochs or 20,
        "batch_size": args.batch_size or 32,
        "train_size": args.train_size or 0.8,
        "class_order": args.class_order,
        "augment": args.augment if args.augment is not None else True,
        "seed": args.seed or 42,
        "verbose": args.verbose
    }

    # Load from config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            
            if "classes" in cfg:
                params["class_order"] = cfg["classes"].get("order", params["class_order"])
            
            if "training" in cfg:
                t_cfg = cfg["training"]
                params["arch"] = t_cfg.get("architecture", params["arch"])
                params["epochs"] = t_cfg.get("epochs", params["epochs"])
                params["batch_size"] = t_cfg.get("batch_size", params["batch_size"])
                params["dropout_rate"] = t_cfg.get("dropout_rate", params.get("dropout_rate", 0.3))
                params["fine_tune"] = t_cfg.get("fine_tune", params.get("fine_tune", False))
                params["train_size"] = t_cfg.get("train_size", params["train_size"])
                params["augment"] = t_cfg.get("augment", params["augment"])
                params["seed"] = t_cfg.get("seed", params["seed"])

    # Overwrite with any explicitly provided CLI arguments
    if args.data_dir: params["data_dir"] = args.data_dir
    if args.output_dir: params["output_dir"] = args.output_dir
    if args.arch: params["arch"] = args.arch
    if args.epochs: params["epochs"] = args.epochs
    if args.dropout_rate: params["dropout_rate"] = args.dropout_rate
    if args.fine_tune: params["fine_tune"] = args.fine_tune
    if args.class_order: params["class_order"] = args.class_order

    # Validate required parameters
    if not params.get("data_dir"):
        print("Error: Missing required argument: --data_dir")
        sys.exit(1)
    if not params.get("class_order"):
        print("Error: Missing required argument: --class_order (or specified in --config)")
        sys.exit(1)

    # Load data
    x_path = os.path.join(params["data_dir"], "X.pkl")
    y_path = os.path.join(params["data_dir"], "Y.pkl")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Data files not found in {params['data_dir']}")
        sys.exit(1)

    print(f"Loading data from {params['data_dir']}...")
    with open(x_path, "rb") as f:
        X = pickle.load(f)
    with open(y_path, "rb") as f:
        Y = pickle.load(f)

    print(f"Initializing Trainer with {len(X)} samples across {len(params['class_order'])} classes...")
    trainer = Trainer(output_dir=params["output_dir"], seed=params["seed"])

    model_path = trainer.train(
        X=X,
        Y=Y,
        class_order=params["class_order"],
        model_architecture=params["arch"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        train_size=params["train_size"],
        augment=params["augment"],
        verbose=params["verbose"],
        dropout_rate=params.get("dropout_rate", 0.3),
        fine_tune=params.get("fine_tune", False)
    )

    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
