"""
annotation_reader.py
Cleaned AnnotationReader for .svl (Sonic Visualiser) and Raven-style annotations.
"""
import os
from typing import Tuple
import pandas as pd
import librosa
import numpy as np
from xml.dom import minidom


class annotation_reader:
    def __init__(self, annotation_file_name: str, base_path: str, file_type: str, audio_extension: str = ".WAV"):
        """
        Parameters
        ----------
        annotation_file_name : str
            Name of annotation file (without extension for svl, or full filename for raven).
        base_path : str
            Path to the species folder that contains 'Audio/' and 'Annotations/' subfolders.
        file_type : str
            'svl' or 'raven_caovitgibbons'
        audio_extension : str
            Audio file extension including dot (e.g. '.WAV', '.wav').
        """
        self.annotation_file_name = annotation_file_name
        self.base_path = base_path
        self.file_type = file_type
        self.audio_extension = audio_extension

    # Audio loader helper
    
    def read_audio_file(self, file_name: str) -> Tuple[np.ndarray, int]:  # type: ignore[name-defined]
        audio_path = os.path.join(self.base_path, "Audio", file_name)
        audio_amps, sr = librosa.load(audio_path, sr=None)
        return audio_amps, sr

    # Public interface
   
    def get_annotation_information(self) -> Tuple[pd.DataFrame, str]:
        """Return a dataframe with columns ['Start','End','Label'] and the audio filename."""
        if self.file_type == "svl":
            return self._read_svl()
        if self.file_type == "raven_caovitgibbons":
            return self._read_raven()
        raise ValueError(f"Unsupported file_type: {self.file_type}")

    # SVL reader
   
    def _read_svl(self) -> Tuple[pd.DataFrame, str]:
        xml_path = os.path.join(self.base_path, "Annotations", self.annotation_file_name + ".svl")
        xmldoc = minidom.parse(xml_path)
        points = xmldoc.getElementsByTagName("point")

        starts, ends, labels = [], [], []
        fname_no_ext = self.annotation_file_name
        audio_file = fname_no_ext + self.audio_extension

        # Ensure audio exists and get sr
        _, sr = self.read_audio_file(audio_file)

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

            # only keep confident annotations (conf==10)
            if conf == 10:
                start_sec = start_frame / sr
                dur_sec = duration_frames / sr
                starts.append(start_sec)
                ends.append(start_sec + dur_sec)
                labels.append(label)

        df = pd.DataFrame({"Start": starts, "End": ends, "Label": labels})
        return df, audio_file
 
    # Raven reader
  
    def _read_raven(self) -> Tuple[pd.DataFrame, str]:
        csv_path = os.path.join(self.base_path, "Annotations", self.annotation_file_name)
        df = pd.read_csv(csv_path, sep="\t")
        out_df = pd.DataFrame({
            "Start": df["Begin Time (s)"],
            "End": df["End Time (s)"],
            "Label": df["Label"]
        })
        audio_file = df["Begin File"].iloc[0]
        return out_df, audio_file

    # Helper for possible nested folders based on filename convention
   
    def get_audio_location(self) -> str:
        if "-" in self.annotation_file_name:
            folder = self.annotation_file_name[: self.annotation_file_name.rfind("-")]
            return folder.replace("-", os.sep) + os.sep
        return ""
