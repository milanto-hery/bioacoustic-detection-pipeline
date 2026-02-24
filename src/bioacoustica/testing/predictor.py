"""
predictor.py
Generalized Predictor for sliding window inference.
"""
import os
import gc
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from scipy import signal
from tensorflow.keras.models import load_model
from yattag import Doc, indent
from tqdm import tqdm


class Predictor:
    def __init__(
        self,
        audio_dir: str,
        output_dir: str,
        model_path: str,
        lowpass_cutoff: int = 3500,
        downsample_rate: int = 8000,
        nyquist_rate: int = 4000,
        segment_duration: int = 4,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        f_min: int = 400,
        f_max: int = 4000,
        audio_extension: str = ".wav",
    ):
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.model_path = model_path
        self.lowpass_cutoff = lowpass_cutoff
        self.downsample_rate = downsample_rate
        self.nyquist_rate = nyquist_rate
        self.segment_duration = segment_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.audio_extension = audio_extension

        self.model = load_model(model_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def create_segments(self, mono_data: np.ndarray, sample_rate: int) -> np.ndarray:
        total_seconds = int(len(mono_data) / sample_rate)
        segments = []
        for start in range(0, max(1, total_seconds - self.segment_duration + 1)):
            s = mono_data[int(start * sample_rate) : int((start + self.segment_duration) * sample_rate)]
            if len(s) == int(self.segment_duration * sample_rate):
                segments.append(s)
        return np.array(segments)

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

    def generate_svl(self, df: pd.DataFrame, sample_rate: int, length_audio_frames: int) -> str:
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

    def predict_file(self, audio_file: str, threshold: float = 0.5):
        audio_path = os.path.join(self.audio_dir, audio_file)
        
        # If file not found, try appending extension
        if not os.path.exists(audio_path):
            if not audio_file.lower().endswith((".wav", ".WAV")):
                alt_path = audio_path + self.audio_extension
                if os.path.exists(alt_path):
                    audio_path = alt_path
                else:
                    # Try toggling extension case
                    ext = ".WAV" if self.audio_extension == ".wav" else ".wav"
                    alt_path = audio_path + ext
                    if os.path.exists(alt_path):
                        audio_path = alt_path
        
        # Final check
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_amps, original_sr = librosa.load(audio_path, sr=None)
        
        # Use actual nyquist rate of the loaded audio
        nyq = original_sr / 2.0
        filtered = self.butter_lowpass_filter(audio_amps, self.lowpass_cutoff, nyq)
        down, sr = self.downsample_file(filtered, original_sr, self.downsample_rate)

        segments = self.create_segments(down, sr)
        
        # Process in batches to handle large files stably
        batch_size = 128
        all_probs = []
        
        if len(segments) > 0:
            for i in tqdm(range(0, len(segments), batch_size), desc=f"Analyzing {audio_file}", leave=False):
                batch_segs = segments[i : i + batch_size]
                specs = self.convert_all_to_image(batch_segs)
                specs = specs[..., np.newaxis]
                
                preds = self.model.predict(specs, verbose=0)
                
                # Assume positive class is at index 1 for binary classification
                if preds.shape[1] >= 2:
                    all_probs.extend(preds[:, 1])
                else:
                    all_probs.extend(preds[:, 0])
        
        probs = np.array(all_probs)
        hits = np.where(probs >= threshold)[0]
        hits = hits.astype(int)

        grouped = self.group_consecutives(hits)
        
        df_vals = []
        for g in grouped:
            start_sec = g[0]
            end_sec = g[-1] + self.segment_duration
            df_vals.append([start_sec, end_sec, 0, self.downsample_rate/2, "predicted"])
        
        df_preds = pd.DataFrame(df_vals, columns=["start(sec)", "end(sec)", "low(freq)", "high(freq)", "label"])

        # Save results
        basename = os.path.splitext(audio_file)[0]
        xml = self.generate_svl(df_preds, original_sr, len(audio_amps))
        output_file = os.path.join(self.output_dir, f"{basename}.svl")
        with open(output_file, "w") as f:
            f.write(xml)

        # Cleanup
        del audio_amps, filtered, down, specs
        gc.collect()
        
        return output_file
