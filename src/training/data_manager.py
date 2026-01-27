from sklearn.model_selection import train_test_split
import numpy as np
import random
from sklearn import utils
from tensorflow.keras.utils import to_categorical


class data_manager:

    def __shuffle_data(self, spectrograms, labels, seed):
        return utils.shuffle(spectrograms, labels, random_state=seed)

    def __augment_with_time_shift(self, X_data, class_label, amount_to_create):
        """Augment spectrograms by random time shifting."""
        augmented_spectrograms = []
        augmented_labels = []

        # determine time axis safely
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

    def __target_encoding(self, y, call_order):
        """Convert string labels to one-hot encoding."""
        y_encoded = np.copy(y)
        for index, call_type in enumerate(call_order):
            y_encoded[y == call_type] = index

        y_encoded = y_encoded.astype(int)
        return to_categorical(y_encoded, num_classes=len(call_order))

    def augment_and_prep_data(
        self,
        presence_class_label,
        X_calls,
        y,
        seed,
        train_size,
        call_order,
        verbose,
    ):

        if len(X_calls) != len(y):
            raise ValueError(
                f"Mismatch: X_calls has {len(X_calls)} samples, y has {len(y)} labels"
            )

        X_calls_train, X_calls_test, y_train, y_test = train_test_split(
            X_calls,
            y,
            shuffle=True,
            random_state=seed,
            train_size=train_size,
        )

        if verbose:
            print("Original split shapes")
            print("Training__: ", X_calls_train.shape)
            print("Validation: ", X_calls_test.shape)

        # --- Identify classes
        presence_idx = np.where(y_train == presence_class_label)[0]
        absence_idx = np.where(y_train != presence_class_label)[0]

        n_presence = len(presence_idx)
        n_absence = len(absence_idx)

        if verbose:
            print(f"Presence: {n_presence}, Absence: {n_absence}")

        # --- Determine minority
        if n_presence == n_absence:
            if verbose:
                print("Classes already balanced, skipping augmentation.")
        else:
            if n_presence < n_absence:
                minority_idx = presence_idx
                minority_label = presence_class_label
                diff = n_absence - n_presence
            else:
                minority_idx = absence_idx
                minority_label = "NOISE"   # explicit as requested
                diff = n_presence - n_absence

            if verbose:
                print(f"Augmenting '{minority_label}' by {diff}")

            X_aug, y_aug = self.__augment_with_time_shift(
                X_calls_train[minority_idx],
                minority_label,
                diff,
            )

            X_calls_train = np.concatenate((X_calls_train, X_aug))
            y_train = np.concatenate((y_train, y_aug))

        # --- Shuffle
        X_calls_train, y_train = self.__shuffle_data(
            X_calls_train, y_train, seed
        )

        # --- Encode
        y_train = self.__target_encoding(y_train, call_order)
        y_test = self.__target_encoding(y_test, call_order)

        if verbose:
            print("Shapes (after augmentation):")
            print("X_calls_train", X_calls_train.shape)
            print("X_calls_test", X_calls_test.shape)
            print("y_train", y_train.shape)
            print("y_test", y_test.shape)

            unique, counts = np.unique(y_train.argmax(axis=1), return_counts=True)
            print("Training distribution:", dict(zip(unique, counts)))

        return X_calls_train, X_calls_test, y_train, y_test
