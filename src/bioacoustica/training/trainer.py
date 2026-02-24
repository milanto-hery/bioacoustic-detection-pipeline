"""
trainer.py
Trainer class for model lifecycle management.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from bioacoustica.training.networks import get_model
from bioacoustica.training.data_manager import DataManager


class Trainer:
    def __init__(self, output_dir: str = "models", seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        self.model = None
        self.history = None
        self.data_manager = DataManager(seed=seed)
        os.makedirs(self.output_dir, exist_ok=True)

    def compile_model(self, model_architecture: str, input_shape: tuple, num_classes: int, **kwargs):
        """
        Uses the Model Factory to instantiate and compile the desired architecture.
        """
        self.model = get_model(
            model_name=model_architecture,
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )
        return self.model

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        class_order: list,
        model_architecture: str = "custom_cnn",
        epochs: int = 20,
        batch_size: int = 32,
        train_size: float = 0.8,
        augment: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        # Prepare data
        X_train, X_val, y_train, y_val = self.data_manager.prepare_data(
            X, Y, class_order, train_size=train_size, augment=augment, verbose=verbose
        )

        # Ensure channel dimension
        if X_train.ndim == 3:
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]

        input_shape = X_train.shape[1:]
        num_classes = len(class_order)

        self.compile_model(model_architecture, input_shape, num_classes, **kwargs)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_architecture}_{timestamp}"
        filepath = os.path.join(self.output_dir, f"{model_name}_best.h5")

        callbacks = [
            ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if verbose else 0,
        )

        final_path = os.path.join(self.output_dir, f"{model_name}_final.h5")
        self.model.save(final_path)
        
        return final_path

    def plot_history(self, save_path: Optional[str] = None):
        if self.history is None:
            return

        h = self.history.history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(h["loss"], label="Train Loss")
        plt.plot(h["val_loss"], label="Val Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(h["accuracy"], label="Train Acc")
        plt.plot(h["val_accuracy"], label="Val Acc")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_confusion_matrix(self, X_val, y_val, class_names, save_path: Optional[str] = None):
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
