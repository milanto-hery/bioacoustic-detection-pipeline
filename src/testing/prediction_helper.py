import os
import time
import glob
import gc
from typing import List, Tuple
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from tensorflow.keras.models import load_model
from yattag import Doc, indent


class prediction_helper:
    def __init__(
        self,
        species_folder: str,
        lowpass_cutoff: int,
        downsample_rate: int,
        nyquist_rate: int,
        segment_duration: int,
        positive_class: str,
        negative_class: str,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: int,
        f_max: int,
        audio_extension: str,
        saved_weights_folder: str,
        testing_list: str = None,
    ):
        self.species_folder = species_folder
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.audio_path = os.path.join(species_folder, "Audio")
        self.annotations_path = os.path.join(species_folder, "Annotations")
        self.saved_data_path = os.path.join(species_folder, "Saved_Data")
        self.testing_files = testing_list or os.path.join(species_folder, "DataFiles", "TestingFiles.txt")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.audio_extension = audio_extension
        self.saved_weights_folder = saved_weights_folder

    # create sliding segments (1s -> list)

    def create_segments(self, mono_data: np.ndarray, time_to_extract: int, sample_rate: int) -> np.ndarray:
        total_seconds = int(len(mono_data) / sample_rate)
        segments = []
        for start in range(0, max(1, total_seconds - time_to_extract + 1)):
            s = mono_data[int(start * sample_rate) : int((start + time_to_extract) * sample_rate)]
            if len(s) == int(time_to_extract * sample_rate):
                segments.append(s)
        return np.array(segments)


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

    # Spectrogram utilities

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

    def add_keras_dim(self, specs: np.ndarray) -> np.ndarray:
        return specs[..., np.newaxis]


    # model loading

    def load_model_weights(self, path: str):
        print(f"Loading model: {path}")
        return load_model(path)
      
    # group helpers
 
    def group_consecutives(self, vals: np.ndarray, step: int = 1) -> List[List[int]]:
        if len(vals) == 0:
            return []
        groups = []
        current = [vals[0]]
        for v in vals[1:]:
            if v == current[-1] + step:
                current.append(v)
            else:
                groups.append(current)
                current = [v]
        groups.append(current)
        return groups

    def group(self, L: List[int]):
        L = sorted(L)
        if not L:
            return []
        first = last = L[0]
        out = []
        for n in L[1:]:
            if n - 1 == last:
                last = n
            else:
                out.append((first, last))
                first = last = n
        out.append((first, last))
        return out
 
    # svl writer

    def dataframe_to_svl(self, df: pd.DataFrame, sample_rate: int, length_audio_frames: int) -> str:
        doc, tag, text = Doc().tagtext()
        doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
        doc.asis('<!DOCTYPE sonic-visualiser>')
        with tag("sv"):
            with tag("data"):
                model_string = (
                    f'<model id="1" name="" sampleRate="{sample_rate}" start="0" end="{length_audio_frames}" '
                    f'type="sparse" dimensions="2" resolution="1" notifyOnAdd="true" dataset="0" subtype="box" '
                    f'minimum="0" maximum="{sample_rate/2}" units="Hz" />'
                )
                doc.asis(model_string)
                with tag("dataset", id="0", dimensions="2"):
                    for _, row in df.iterrows():
                        point = (
                            f'<point frame="{int(row["start(sec)"]*sample_rate)}" value="{int(row.get("low(freq)",0))}" '
                            f'duration="{int((row["end(sec)"]-row["start(sec)"])*sample_rate)}" '
                            f'extent="{int(row.get("high(freq)",0))}" label="{row["label"]}" />'
                        )
                        doc.asis(point)
            with tag("display"):
                display_string = '<layer id="2" type="boxes" name="Boxes" model="1"  verticalScale="0"  colourName="White" colour="#ffffff" darkBackground="true" />'
                doc.asis(display_string)
        return indent(doc.getvalue(), indentation=" " * 2, newline="\r\n")

    # main predict loop

    def predict_all_test_files(self, weights_folder: str, saved_prediction: str):
        testing_files = pd.read_csv(self.testing_files, header=None)[0].tolist()
        os.makedirs(saved_prediction, exist_ok=True)

        for test_file in testing_files:
            fname = test_file
            print(f"Processing: {fname}")

            audio_path = os.path.join(self.audio_path, fname + self.audio_extension)
            if not os.path.exists(audio_path):
                print(f"Missing audio: {audio_path}")
                continue

            audio_amps, original_sr = self.read_audio_file(audio_path)
            filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, self.nyquist_rate)
            down, sr = self.downsample_file(filtered, original_sr, self.downsample_rate)

            segments = self.create_segments(down, self.segment_duration, sr)
            specs = self.convert_all_to_image(segments)
            specs = self.add_keras_dim(specs)

            model_files = [os.path.join(weights_folder, f) for f in os.listdir(weights_folder) if f.endswith(".hdf5") or f.endswith(".keras")]
            for model_file in model_files:
                model = self.load_model_weights(model_file)
                preds = model.predict(specs)
                # take probability of positive class at index 1
                if preds.ndim == 2 and preds.shape[1] >= 2:
                    probs = preds[:, 1]
                else:
                    probs = preds[:, 0]

                hits = np.where(probs >= 0.5)[0]
                hits = hits.astype(int)

                grouped = self.group_consecutives(hits)
                predictions = []
                for g in grouped:
                    predictions.extend(g)

                if predictions:
                    predicted_groups = list(self.group(predictions))
                    df_vals = []
                    for (s, e) in predicted_groups:
                        df_vals.append([s, e + self.segment_duration, 900, 1600, "predicted"])
                    df_preds = pd.DataFrame(df_vals, columns=["start(sec)", "end(sec)", "low(freq)", "high(freq)", "label"])

                    xml = self.dataframe_to_svl(df_preds, original_sr, len(audio_amps))
                    outdir = os.path.join(saved_prediction, os.path.basename(model_file).split(".")[0])
                    os.makedirs(outdir, exist_ok=True)
                    with open(os.path.join(outdir, fname + ".svl"), "w") as f:
                        f.write(xml)

            # cleanup
            del audio_amps, filtered, down, specs
            gc.collect()
            time.sleep(0.5)

        return True
