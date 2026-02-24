"""
evaluator.py
Evaluator for event-based metrics.
"""
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import librosa
from xml.dom import minidom
from datetime import datetime, timedelta


class Evaluator:
    def __init__(self, audio_dir: str):
        self.audio_dir = audio_dir

    def read_audio_metadata(self, audio_file: str):
        audio_path = os.path.join(self.audio_dir, audio_file)
        sr = librosa.get_samplerate(audio_path)
        duration = librosa.get_duration(path=audio_path)
        return sr, duration

    def read_svl(self, svl_path: str, audio_file: str):
        xmldoc = minidom.parse(svl_path)
        points = xmldoc.getElementsByTagName("point")

        sr, _ = self.read_audio_metadata(audio_file)
        
        # Base datetime for overlap calculations if timestamps are needed
        # Fallback to Unix epoch if filename doesn't contain a date
        try:
             # Try to extract date from filename (HGSM3AC_0+1_20150616_050750.wav)
             base_name = os.path.basename(audio_file)
             date_str = base_name[base_name.index('+1') + 3: base_name.rfind(".")]
             base_dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        except:
             base_dt = datetime(1970, 1, 1)

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
                "Label": label,
                "StartSec": start_sec,
                "EndSec": start_sec + dur_sec,
                "AudioFile": audio_file
            })

        return pd.DataFrame(rows)

    def total_audio_hours(self, file_list: Optional[List[str]] = None):
        """
        Calculate total hours of audio. If file_list is provided, only count those files.
        """
        total_sec = 0.0
        if file_list:
            files_to_scan = []
            for f in file_list:
                # Handle both base names and full filenames
                if not f.lower().endswith((".wav", ".WAV")):
                    files_to_scan.append(f + ".wav")
                else:
                    files_to_scan.append(f)
        else:
            files_to_scan = [f for f in os.listdir(self.audio_dir) if f.lower().endswith((".wav", ".WAV"))]

        for f in files_to_scan:
            audio_path = os.path.join(self.audio_dir, f)
            if not os.path.exists(audio_path):
                # Try uppercase extension
                if f.endswith(".wav"): audio_path = os.path.join(self.audio_dir, f.replace(".wav", ".WAV"))
                elif f.endswith(".WAV"): audio_path = os.path.join(self.audio_dir, f.replace(".WAV", ".wav"))
            
            if os.path.exists(audio_path):
                try:
                    _, duration = self.read_audio_metadata(os.path.basename(audio_path))
                    total_sec += duration
                except Exception as e:
                    print(f"Warning: Could not read {audio_path}: {e}")
        
        return total_sec / 3600.0

    def merge_events(self, df: pd.DataFrame, gap_tolerance: float = 0.5):
        """
        Merge events that are closer than gap_tolerance seconds.
        """
        if df.empty:
            return df

        # Ensure sorted by start time
        df = df.sort_values("StartSec").reset_index(drop=True)
        
        merged = []
        if len(df) > 0:
            curr = df.iloc[0].to_dict()
            
            for i in range(1, len(df)):
                next_event = df.iloc[i]
                
                # If overlap or gap within tolerance, merge
                if next_event["StartSec"] <= curr["EndSec"] + gap_tolerance:
                    curr["EndSec"] = max(curr["EndSec"], next_event["EndSec"])
                    # Start/End timestamps handle the datetime aspect
                    duration = curr["EndSec"] - curr["StartSec"]
                    curr["End"] = curr["Start"] + timedelta(seconds=duration)
                else:
                    merged.append(curr)
                    curr = next_event.to_dict()
            merged.append(curr)
            
        return pd.DataFrame(merged)

    def evaluate(
        self,
        df_gt: pd.DataFrame,
        df_pred: pd.DataFrame,
        target_label: str,
        min_overlap_pct: float = 0.5,
        merge_tolerance: float = 0.5,
        test_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Event-based evaluation with detection merging.
        Uses absolute timestamps to ensure correct matching across multiple files.
        If test_files is provided, it uses it for total time calculation (handles zero-detection files).
        """
        if df_gt.empty:
            return {"Error": "Ground truth is empty"}
            
        # 1. Filter and Merge
        df_gt_target = df_gt[df_gt["Label"] == target_label].copy()
        
        # In our pipeline, detections are labeled 'predicted' or follow target_label
        # We allow both for flexibility, but mostly we want to match whatever is in df_pred
        df_pred_target = df_pred.copy()
        
        # Merge fragmented predictions (critical for segment-based inference)
        # We merge based on absolute time to handle gaps correctly
        df_pred_merged = self.merge_events_dt(df_pred_target, gap_tolerance=merge_tolerance)
        
        matched_gt_idx = set()
        matched_pred_idx = set()

        # 2. Match Predictions to GT using absolute Datetimes
        for pred_idx, pred in df_pred_merged.iterrows():
            overlaps = df_gt_target[
                (df_gt_target["Start"] < pred["End"]) &
                (df_gt_target["End"] > pred["Start"])
            ]
            
            for gt_idx, gt in overlaps.iterrows():
                latest_start = max(gt["Start"], pred["Start"])
                earliest_end = min(gt["End"], pred["End"])
                overlap_sec = (earliest_end - latest_start).total_seconds()
                gt_duration = (gt["End"] - gt["Start"]).total_seconds()
                
                # Check if overlap meets the threshold (50% of the GT duration)
                if overlap_sec / gt_duration >= min_overlap_pct:
                    matched_pred_idx.add(pred_idx)
                    matched_gt_idx.add(gt_idx)

        TP_recall = len(matched_gt_idx)
        TP_precision = len(matched_pred_idx)
        
        FN = len(df_gt_target) - TP_recall
        FP = len(df_pred_merged) - TP_precision

        precision = TP_precision / len(df_pred_merged) if len(df_pred_merged) > 0 else 0.0
        recall = TP_recall / len(df_gt_target) if len(df_gt_target) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # We calculate FP/hr based on the total hours of the FILES UNDER TEST
        # If test_files is provided, use it (most accurate). Else fallback to detected files.
        if test_files:
            total_hours = self.total_audio_hours(test_files)
        else:
            total_hours = self.total_audio_hours(df_pred["AudioFile"].unique().tolist()) if "AudioFile" in df_pred.columns else self.total_audio_hours()
        
        fp_per_hour = FP / total_hours if total_hours > 0 else np.nan

        return {
            "TP": TP_recall,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FP_per_hour": fp_per_hour,
            "TotalHours": total_hours,
            "NumGT": len(df_gt_target),
            "NumDetections": len(df_pred_merged)
        }

    def merge_events_dt(self, df: pd.DataFrame, gap_tolerance: float = 0.5):
        """
        Merge events using absolute datetimes.
        """
        if df.empty:
            return df

        df = df.sort_values("Start").reset_index(drop=True)
        
        merged = []
        if len(df) > 0:
            curr = df.iloc[0].to_dict()
            
            for i in range(1, len(df)):
                next_event = df.iloc[i]
                
                # Use absolute time difference
                gap = (next_event["Start"] - curr["End"]).total_seconds()
                
                if gap <= gap_tolerance:
                    curr["End"] = max(curr["End"], next_event["End"])
                else:
                    merged.append(curr)
                    curr = next_event.to_dict()
            merged.append(curr)
            
        return pd.DataFrame(merged)
