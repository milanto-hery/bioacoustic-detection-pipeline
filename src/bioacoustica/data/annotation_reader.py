"""
annotation_reader.py
Generalized AnnotationReader for .svl (Sonic Visualiser) and Raven-style annotations.
"""
import os
from typing import Tuple
import pandas as pd
import librosa
import numpy as np
from xml.dom import minidom


class AnnotationReader:
    def __init__(self, annotation_file: str, audio_dir: str, annotations_dir: str, file_type: str, audio_extension: str = ".wav"):
        """
        Parameters
        ----------
        annotation_file : str
            Full path or filename of the annotation file.
        audio_dir : str
            Path to the directory containing audio files.
        annotations_dir : str
            Path to the directory containing annotation files.
        file_type : str
            'svl' or 'raven'
        audio_extension : str
            Audio file extension including dot (e.g. '.wav').
        """
        self.annotation_file = annotation_file
        self.audio_dir = audio_dir
        self.annotations_dir = annotations_dir
        self.file_type = file_type
        self.audio_extension = audio_extension

    def get_samplerate(self, file_name: str) -> int:
        audio_path = os.path.join(self.audio_dir, file_name)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return librosa.get_samplerate(audio_path)

    def get_annotation_information(self) -> Tuple[pd.DataFrame, str]:
        """Return a dataframe with columns ['Start','End','Label'] and the audio filename."""
        if self.file_type == "svl":
            return self._read_svl()
        if self.file_type == "raven":
            return self._read_raven()
        raise ValueError(f"Unsupported file_type: {self.file_type}")

    def _read_svl(self) -> Tuple[pd.DataFrame, str]:
        # If annotation_file is just a name, append .svl and join with directory
        if not os.path.isabs(self.annotation_file) and not self.annotation_file.endswith(".svl"):
             xml_path = os.path.join(self.annotations_dir, self.annotation_file + ".svl")
        else:
             xml_path = self.annotation_file
             
        xmldoc = minidom.parse(xml_path)
        points = xmldoc.getElementsByTagName("point")

        starts, ends, labels = [], [], []
        # Audio file name is usually the same as the svl file name
        fname_no_ext = os.path.splitext(os.path.basename(xml_path))[0]
        audio_file = fname_no_ext + self.audio_extension

        sr = self.get_samplerate(audio_file)

        for p in points:
            start_frame = float(p.getAttribute("frame"))
            duration_frames = float(p.getAttribute("duration"))
            raw_label = p.getAttribute("label")

            if raw_label == "":
                continue

            # parse optional confidence e.g. "SPECIES,10"
            if "," in raw_label:
                label_str, conf_str = raw_label.split(",", 1)
                try:
                    conf = int(conf_str)
                except ValueError:
                    conf = 10
                label = label_str.strip()
            else:
                label = raw_label.strip()
                conf = 10

            if conf == 10:
                start_sec = start_frame / sr
                dur_sec = duration_frames / sr
                starts.append(start_sec)
                ends.append(start_sec + dur_sec)
                labels.append(label)

        df = pd.DataFrame({"Start": starts, "End": ends, "Label": labels})
        return df, audio_file

    def _read_raven(self) -> Tuple[pd.DataFrame, str]:
        csv_path = self.annotation_file if os.path.isabs(self.annotation_file) else os.path.join(self.annotations_dir, self.annotation_file)
        df = pd.read_csv(csv_path, sep="\t")
        out_df = pd.DataFrame({
            "Start": df["Begin Time (s)"],
            "End": df["End Time (s)"],
            "Label": df["Label"]
        })
        audio_file = df["Begin File"].iloc[0]
        return out_df, audio_file
