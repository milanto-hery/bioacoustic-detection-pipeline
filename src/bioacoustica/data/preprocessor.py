"""
preprocessor.py
Generalized Preprocessor for feature extraction and dataset creation.
"""
import os
import pickle
from typing import Tuple, List, Union
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from tqdm import tqdm

from bioacoustica.data.annotation_reader import AnnotationReader

Array = np.ndarray


class Preprocessor:
    def __init__(
        self,
        audio_dir: str,
        annotations_dir: str,
        output_dir: str,
        lowpass_cutoff: int = 3500,
        downsample_rate: int = 8000,
        nyquist_rate: int = 4000,
        segment_duration: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: int = 400,
        f_max: int = 4000,
    ):
        self.audio_dir = audio_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        os.makedirs(self.output_dir, exist_ok=True)

    def read_audio_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr

    def butter_lowpass_filter(self, data: np.ndarray, cutoff_freq: float, nyq_freq: float, order: int = 4) -> np.ndarray:
        normal_cutoff = cutoff_freq / nyq_freq
        b, a = signal.butter(order, normal_cutoff, btype="lowpass")
        y = signal.filtfilt(b, a, data)
        return y

    def downsample_file(self, amplitudes: np.ndarray, original_sr: int, new_sample_rate: int) -> Tuple[np.ndarray, int]:
        res = librosa.resample(amplitudes, orig_sr=original_sr, target_sr=new_sample_rate, res_type="soxr_hq")
        return res, new_sample_rate

    def convert_single_to_image(self, audio: np.ndarray) -> np.ndarray:
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.downsample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels, 
            fmin=self.f_min, 
            fmax=self.f_max
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        vmin = -80
        S_db = np.clip(S_db, vmin, 0)
        S_scaled = (S_db - vmin) / (0 - vmin)
        return S_scaled.astype(np.float32)

    def convert_all_to_image(self, segments: List[np.ndarray]) -> np.ndarray:
        return np.array([self.convert_single_to_image(seg) for seg in segments], dtype=np.float32)

    def get_segments(self, audio_amplitudes: Array, start_sec: float, duration_seconds: float, label: str, background_classes: List[str] = None):
        X_segments = []
        Y_labels = []

        if duration_seconds < self.segment_duration:
            segments_to_extract = 1
        else:
            segments_to_extract = int(duration_seconds - self.segment_duration + 1)

        if background_classes and label in background_classes:
            segments_to_extract = min(segments_to_extract, 10)

        for i in range(segments_to_extract):
            start_idx = int((start_sec + i) * self.downsample_rate)
            end_idx = start_idx + int(self.downsample_rate * self.segment_duration)

            if end_idx > len(audio_amplitudes):
                continue

            segment = audio_amplitudes[start_idx:end_idx]
            X_segments.append(segment)
            Y_labels.append(label)

        return X_segments, Y_labels

    def save_dataset(self, X: Array, Y: Array, filename_prefix: str = ""):
        with open(os.path.join(self.output_dir, f"{filename_prefix}X.pkl"), "wb") as f:
            pickle.dump(X, f, protocol=4)
        with open(os.path.join(self.output_dir, f"{filename_prefix}Y.pkl"), "wb") as f:
            pickle.dump(Y, f, protocol=4)

    def load_dataset(self, filename_prefix: str = "") -> Tuple[Array, Array]:
        with open(os.path.join(self.output_dir, f"{filename_prefix}X.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(os.path.join(self.output_dir, f"{filename_prefix}Y.pkl"), "rb") as f:
            Y = pickle.load(f)
        return X, Y

    def create_dataset_from_files(self, file_list: List[str], file_type: str, audio_extension: str = ".wav", background_classes: List[str] = None, verbose: bool = False):
        X_all = []
        Y_all = []

        for file_name in tqdm(file_list, desc="Processing files", disable=not verbose):
            # Try to get the actual audio filename
            # The current logic in preprocessing.py was a bit messy with raven naming
            # We will use the AnnotationReader to help
            
            # If it's a raven file, the file_name is the annotation file
            # If it's svl, file_name is usually the audio basename
            
            if verbose:
                print(f"Processing: {file_name}")

            try:
                reader = AnnotationReader(
                    annotation_file=file_name,
                    audio_dir=self.audio_dir,
                    annotations_dir=self.annotations_dir,
                    file_type=file_type,
                    audio_extension=audio_extension
                )
                df, audio_file = reader.get_annotation_information()
            except Exception as e:
                if verbose:
                    print(f"Error reading annotations for {file_name}: {e}")
                continue

            audio_path = os.path.join(self.audio_dir, audio_file)
            if not os.path.exists(audio_path):
                if verbose:
                    print(f"Audio missing: {audio_path}")
                continue

            audio_amps, original_sr = self.read_audio_file(audio_path)
            nyq = original_sr / 2.0
            filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, nyq)
            amplitudes, _ = self.downsample_file(filtered, original_sr, self.downsample_rate)

            for _, row in df.iterrows():
                start_seconds = row["Start"]
                end_seconds = row["End"]
                label = row["Label"]
                duration = end_seconds - start_seconds

                X_data, y_data = self.get_segments(amplitudes, start_seconds, duration, label, background_classes)
                if len(X_data) == 0:
                    continue

                X_specs = self.convert_all_to_image(X_data)
                X_all.extend(X_specs)
                Y_all.extend(y_data)

        X_all = np.asarray(X_all, dtype=np.float32)
        Y_all = np.asarray(Y_all, dtype=object)

        return X_all, Y_all
