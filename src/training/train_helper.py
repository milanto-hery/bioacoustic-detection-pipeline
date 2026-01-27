import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

from .cnn_network import build_baseline_cnn, build_resnet50, build_mobilenet_v2
from .data_manager import data_manager


class train_helper:
    def __init__(self, species_folder):
        self.model = None
        self.history = None
        self.species_folder = species_folder
        self.params = self.load_json_config(self.species_folder, "Params/params.json")
        self.data_manager = data_manager()


    def load_json_config(self, species_folder: str, file_name: str) -> dict:
        file_path = os.path.join(species_folder, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config not found: {file_path}")
        with open(file_path, "r") as f:
            return json.load(f)

    def compile_model(self, model_name: str, input_shape, num_classes: int):
        if model_name.lower() in ["cnn", "cnn_2", "baseline_cnn"]:
            self.model = build_baseline_cnn(input_shape, num_classes=num_classes)
        elif model_name.lower() == "resnet50":
            self.model = build_resnet50(input_shape, num_classes=num_classes)
        elif model_name.lower() == "mobilenetv2":
            self.model = build_mobilenet_v2(input_shape, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.model.summary()

    def train(self, X: np.ndarray, Y: np.ndarray, species_key: str, model_name: str, runs: int = 1):
        cfg = self.params[species_key]
        call_order = cfg["call_order"]
        absence_label = cfg["absence_class_label"]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]
        train_size = cfg["train_size"]
        seed = cfg["seed"]
        verbose = cfg.get("verbose", True)

        # add one channel dimension if missing
        if len(X.shape) == 3:
            X = X[..., np.newaxis]

        # If data has no channel dimension, keep as is (DataManager may expect HxW)
        X_train, X_val, y_train, y_val = self.data_manager.augment_and_prep_data(
            absence_label, X, Y, seed, train_size, call_order, verbose
        )

        # infer input shape for model
        input_shape = X_train.shape[1:]
        num_classes = y_train.shape[1]

        self.compile_model(model_name, input_shape, num_classes)

        save_dir = "Saved_weights"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"{model_name}_{timestamp}_best.keras")

        callbacks = [
            ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        
        self.plot_history()
        
        self.plot_confusion_matrix(X_val, y_val)

        # save final model
        final_path = os.path.join(save_dir, f"{model_name}_{timestamp}_final.keras")
        self.model.save(final_path)
        if verbose:
            print(f"Saved final model to {final_path}")

        return final_path

    def plot_history(self):
        if self.history is None:
            print("No history to plot.")
            return

        h = self.history.history
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(h["loss"], label="train_loss")
        plt.plot(h["val_loss"], label="val_loss")
        plt.legend()
        plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(h["accuracy"], label="train_acc")
        plt.plot(h["val_accuracy"], label="val_acc")
        plt.legend()
        plt.title("Accuracy")
        plt.show()
        
    def plot_confusion_matrix(self, X_val, y_val):
        # Predict the labels for the validation set
        y_pred = self.model.predict(X_val)
        
        # Convert the multilabel-indicator format to binary labels
        y_pred_binary = np.argmax(y_pred, axis=1)
        y_val_binary = np.argmax(y_val, axis=1)
        
        # Compute the confusion matrix
        cm = confusion_matrix(y_val_binary, y_pred_binary)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
