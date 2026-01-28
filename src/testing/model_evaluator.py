
import os
import numpy as np
import pandas as pd
import librosa
from xml.dom import minidom
from datetime import datetime, timedelta


class model_evaluator:

    def filename_to_datetime(self, filename):
        """
        Example filename:
        HGSM3AC_0+1_20150616_050750.wav
        """
        # remove extension if present
        base_name = os.path.basename(filename)
        if base_name.endswith(".wav"):
            base_name = base_name[:-4]
    
        date_str = base_name[base_name.index('+1') + 3:]
        return datetime.strptime(date_str, "%Y%m%d_%H%M%S")

    # --------------------------------------------------
    # read audio
    # --------------------------------------------------
    def read_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        return y, sr

    # --------------------------------------------------
    # read SVL file → DataFrame
    # --------------------------------------------------
    def read_svl(self, svl_path, audio_path):
        xmldoc = minidom.parse(svl_path)
        points = xmldoc.getElementsByTagName("point")

        y, sr = self.read_audio(audio_path)
        base_dt = self.filename_to_datetime(os.path.basename(audio_path))

        rows = []

        for p in points:
            label = p.getAttribute("label")
            if label == "":
                continue

            frame = float(p.getAttribute("frame"))
            duration = float(p.getAttribute("duration"))

            start_sec = frame / sr
            dur_sec = duration / sr

            start_time = base_dt + timedelta(seconds=start_sec)
            end_time = start_time + timedelta(seconds=dur_sec)

            rows.append({
                "Start": start_time,
                "End": end_time,
                "Label": label
            })

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # load all GT + predictions
    # --------------------------------------------------
    def load_all(self, audio_folder, gt_folder, pred_folder):
        df_gt, df_pred = [], []

        for f in os.listdir(audio_folder):
            if not f.endswith(".wav"):
                continue
            name = f[:-4]
            audio_path = os.path.join(audio_folder, f)
            gt_path = os.path.join(gt_folder, name + ".svl")
            if os.path.exists(gt_path):
                df_gt.append(self.read_svl(gt_path, audio_path))

        for f in os.listdir(pred_folder):
            if f.endswith(".svl"):
                audio_path = os.path.join(audio_folder, f.replace(".svl", ".wav"))
                if os.path.exists(audio_path):
                    df_pred.append(self.read_svl(os.path.join(pred_folder, f), audio_path))

        return pd.concat(df_gt, ignore_index=True), pd.concat(df_pred, ignore_index=True)

    # --------------------------------------------------
    # total audio duration (hours)
    # --------------------------------------------------
    def total_audio_hours(self, audio_folder):
        total_sec = 0.0
        for f in os.listdir(audio_folder):
            if f.endswith(".wav"):
                y, sr = librosa.load(os.path.join(audio_folder, f), sr=None)
                total_sec += len(y) / sr
        return total_sec / 3600.0

    # --------------------------------------------------
    # MAIN evaluation using percentage overlap
    # --------------------------------------------------
    def evaluate(
        self,
        df_gt,
        df_pred,
        audio_folder,
        target_label,
        min_overlap_pct=0.5  # TP if overlap >= 50% of GT duration
    ):
        TP = 0
        matched_pred_idx = set()

        # keep only target species in GT
        df_gt = df_gt[df_gt["Label"] == target_label]

        for gt_idx, gt in df_gt.iterrows():
            # find overlapping predictions
            overlaps = df_pred[
                (df_pred["Start"] <= gt["End"]) &
                (df_pred["End"] >= gt["Start"])
            ]

            # compute percentage overlap
            pct_overlap = []
            for pred_idx, pred in overlaps.iterrows():
                latest_start = max(gt["Start"], pred["Start"])
                earliest_end = min(gt["End"], pred["End"])
                overlap_sec = (earliest_end - latest_start).total_seconds()
                gt_duration = (gt["End"] - gt["Start"]).total_seconds()
                pct_overlap.append((pred_idx, overlap_sec / gt_duration))

            # keep only predictions with enough overlap
            matched = [idx for idx, pct in pct_overlap if pct >= min_overlap_pct]

            if matched:
                TP += 1
                matched_pred_idx.update(matched)

        FN = len(df_gt) - TP
        FP = len(df_pred) - len(matched_pred_idx)

        # metrics
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

        # false alarms per hour
        total_hours = self.total_audio_hours(audio_folder)
        FP_per_hour = FP / total_hours if total_hours > 0 else np.nan

        # print results
        print("=" * 45)
        print(" EVENT-BASED DETECTION EVALUATION (percentage overlap) ")
        print("=" * 45)
        print(f"Target species        : {target_label}")
        print(f"True events (GT)      : {len(df_gt)}")
        print(f"True Positives (TP)   : {TP}")
        print(f"False Negatives (FN)  : {FN}")
        print(f"False Positives (FP)  : {FP}")
        print("-" * 45)
        print(f"Precision             : {Precision:.3f}")
        print(f"Recall                : {Recall:.3f}")
        print(f"F1-score              : {F1:.3f}")
        print(f"False alarms / hour   : {FP_per_hour:.2f}")
        print("=" * 45)

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "FP_per_hour": FP_per_hour
        }
