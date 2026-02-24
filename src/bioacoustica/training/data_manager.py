"""
data_manager.py
DataManager for data splitting, augmentation, and encoding.
"""
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import utils
from tensorflow.keras.utils import to_categorical


class DataManager:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def shuffle_data(self, spectrograms, labels):
        return utils.shuffle(spectrograms, labels, random_state=self.seed)

    def augment_with_time_shift(self, X_data, class_label, amount_to_create):
        """Augment spectrograms by random time shifting."""
        augmented_spectrograms = []
        augmented_labels = []

        # determine time axis safely (expecting HxWxC or HxW)
        time_axis = 1 if X_data.ndim == 3 else 2

        for _ in range(amount_to_create):
            idx = random.randint(0, len(X_data) - 1)
            spec = X_data[idx]

            max_shift = max(1, spec.shape[time_axis] // 10)
            shift = random.randint(1, max_shift)

            shifted = np.roll(spec, shift, axis=time_axis)

            augmented_spectrograms.append(shifted)
            augmented_labels.append(class_label)

        return np.asarray(augmented_spectrograms), np.asarray(augmented_labels)

    def target_encoding(self, y, class_order):
        """Convert string labels to one-hot encoding."""
        y_encoded = np.zeros_like(y, dtype=int)
        for index, class_name in enumerate(class_order):
            y_encoded[y == class_name] = index

        return to_categorical(y_encoded, num_classes=len(class_order))

    def prepare_data(
        self,
        X,
        Y,
        class_order,
        train_size=0.8,
        augment=True,
        minority_class=None,
        verbose=True,
    ):
        """
        Split, balance (optional), and encode data.
        """
        if len(X) != len(Y):
            raise ValueError(f"Mismatch: X has {len(X)} samples, Y has {len(Y)} labels")

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            Y,
            shuffle=True,
            random_state=self.seed,
            train_size=train_size,
        )

        if verbose:
            print(f"Original split: Train={X_train.shape}, Val={X_val.shape}")

        if augment:
            # Augment training data to balance classes or just increase variance
            unique, counts = np.unique(y_train, return_counts=True)
            max_count = np.max(counts)
            
            X_aug_list = [X_train]
            y_aug_list = [y_train]
            
            for class_name in class_order:
                # Find current samples for this class
                idx = np.where(y_train == class_name)[0]
                n_current = len(idx)
                
                # If we want to reach max_count or just add a fixed percentage
                # Original logic balanced to the majority class
                n_to_add = max_count - n_current
                
                # If we are already balanced, maybe add a small random percentage for more robustness
                if n_to_add == 0:
                    n_to_add = int(n_current * 0.2) # Add 20% variance

                if n_to_add > 0:
                    if verbose:
                        print(f"Augmenting '{class_name}' by {n_to_add} samples...")
                    
                    X_aug, y_aug = self.augment_with_time_shift(X_train[idx], class_name, n_to_add)
                    X_aug_list.append(X_aug)
                    y_aug_list.append(y_aug)
            
            X_train = np.concatenate(X_aug_list)
            y_train = np.concatenate(y_aug_list)
            
            # Reshuffle
            X_train, y_train = self.shuffle_data(X_train, y_train)

        # One-hot encoding
        y_train_enc = self.target_encoding(y_train, class_order)
        y_val_enc = self.target_encoding(y_val, class_order)

        if verbose:
            print(f"Final shapes: X_train={X_train.shape}, y_train={y_train_enc.shape}")

        return X_train, X_val, y_train_enc, y_val_enc
