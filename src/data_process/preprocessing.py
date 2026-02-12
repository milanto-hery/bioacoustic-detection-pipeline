import os
import pickle
from typing import Tuple, List
import numpy as np
import pandas as pd
import librosa
from scipy import signal

# For typing convenience
Array = np.ndarray


class preprocessing:
    def __init__(
        self,
        species_folder: str,
        lowpass_cutoff: int,
        downsample_rate: int,
        nyquist_rate: int,
        segment_duration: int,
        positive_class: str,
        background_class: List[str],
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        file_type: str,
        audio_extension: str = ".wav",
    ):
        self.species_folder = species_folder
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.background_class = set(background_class)
        self.audio_path = os.path.join(species_folder, "Audio")
        self.annotations_path = os.path.join(species_folder, "Annotations")
        self.saved_data_path = os.path.join(species_folder, "Saved_Data")
        self.training_files = os.path.join(species_folder, "DataFiles", "TrainingFiles.txt")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.file_type = file_type
        self.audio_extension = audio_extension

        os.makedirs(self.saved_data_path, exist_ok=True)

    # Audio utilities
    
    def read_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        # sr=None preserves original 9600Hz
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr

    def butter_lowpass_filter(self, data: np.ndarray, cutoff_freq: float, nyq_freq: float, order: int = 4) -> np.ndarray:
        # Crucial: Ensure cutoff < nyq to avoid error
        normal_cutoff = cutoff_freq / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes: np.ndarray, original_sr: int, new_sample_rate: int) -> Tuple[np.ndarray, int]:
        # Using soxr_hq for better signal preservation than kaiser_fast
        res = librosa.resample(amplitudes, orig_sr=original_sr, target_sr=new_sample_rate, res_type="soxr_hq")
        return res, new_sample_rate

    # Spectrogram Utilities

    def convert_single_to_image(self, audio: np.ndarray) -> np.ndarray:
        # 1. Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.downsample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels, 
            fmin=self.f_min, 
            fmax=self.f_max
        )

        # 2. Convert to Log Scale (Decibels)
        #ref=np.max ensures the brightest point is 0dB
        S_db = librosa.power_to_db(S, ref=np.max)

        # 3. Robust Normalization (Better than Z-score for noisy audio)
        # We clip the range to -80dB to 0dB to remove extreme floor noise
        vmin = -80
        S_db = np.clip(S_db, vmin, 0)
        
        # Scale to 0.0 - 1.0
        S_scaled = (S_db - vmin) / (0 - vmin)
        
        return S_scaled.astype(np.float32)

    def convert_all_to_image(self, segments: List[np.ndarray]) -> np.ndarray:
        return np.array([self.convert_single_to_image(seg) for seg in segments], dtype=np.float32)

    def add_extra_dim(self, spectrograms: Array) -> Array:
        return spectrograms.reshape((spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1))

    # Segmentation
    
    def getXY(self, audio_amplitudes: Array, start_sec: int, annotation_duration_seconds: int, label: str, verbose: bool = False):
        X_segments = []
        Y_labels = []

        if annotation_duration_seconds - self.segment_duration < 0:
            segments_to_extract = 1
        else:
            segments_to_extract = annotation_duration_seconds - self.segment_duration + 1

        if label in self.background_class:
            segments_to_extract = min(segments_to_extract, 10)

        for i in range(int(segments_to_extract)):
            start_idx = int((start_sec + i) * self.downsample_rate)
            end_idx = start_idx + int(self.downsample_rate * self.segment_duration)

            if end_idx > len(audio_amplitudes):
                continue

            segment = audio_amplitudes[start_idx:end_idx]
            X_segments.append(segment)
            Y_labels.append(label)

        return X_segments, Y_labels

    # Saving / loading
    
    def save_spec_data_to_pickle(self, X: Array, Y: Array):
        with open(os.path.join(self.saved_data_path, "X_S.pkl"), "wb") as f:
            pickle.dump(X, f, protocol=4)
        with open(os.path.join(self.saved_data_path, "Y_S.pkl"), "wb") as f:
            pickle.dump(Y, f, protocol=4)

    def load_spec_data_from_pickle(self) -> Tuple[Array, Array]:
        with open(os.path.join(self.saved_data_path, "X_S.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(self.saved_data_path, "Y_S.pkl"), "rb") as f:
            Y = pickle.load(f)
        return X, Y

    # Pipeline: create dataset from training list

    def create_dataset(self, verbose: bool = False):
        X_calls = []
        Y_calls = []

        training_files = pd.read_csv(self.training_files, header=None)[0].tolist()

        for file in training_files:
            if self.file_type == "svl":
                file_name_no_extension = file
            else:
                # for raven-style file names: extract basename without prefix
                file_name_no_extension = file[file.rfind("-") + 1 : file.find(".")]

            if verbose:
                print(f"Processing: {file_name_no_extension}")

            audio_path = os.path.join(self.audio_path, file_name_no_extension + self.audio_extension)
            if not os.path.exists(audio_path):
                if verbose:
                    print(f"Audio missing: {audio_path}")
                continue

            audio_amps, original_sr = self.read_audio_file(audio_path)
            filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)
            amplitudes, sample_rate = self.downsample_file(filtered, original_sr, self.downsample_rate)

            # Use annotation_reader from a separate module to get annotations
            from data_process.annotation_reader import annotation_reader  # local import to avoid circular import
            reader = annotation_reader(file, self.species_folder, self.file_type, self.audio_extension)
            df, audio_file_name = reader.get_annotation_information()

            for _, row in df.iterrows():
                start_seconds = int(round(row["Start"]))
                end_seconds = int(round(row["End"]))
                label = row["Label"]
                duration = end_seconds - start_seconds

                X_data, y_data = self.getXY(amplitudes, start_seconds, duration, label, verbose)
                if len(X_data) == 0:
                    continue

                X_specs = self.convert_all_to_image(X_data)
                X_calls.extend(X_specs)
                Y_calls.extend(y_data)

        X_calls = np.asarray(X_calls, dtype=np.float32)
        Y_calls = np.asarray(Y_calls, dtype=object)  # labels are strings

        return X_calls, Y_calls

    def check_distribution(self, Y):
        unique, counts = np.unique(Y, return_counts=True)
        return dict(zip(unique, counts))
