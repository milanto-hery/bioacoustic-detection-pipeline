
import os
import numpy as np
import pandas as pd
import librosa
from xml.dom import minidom
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class model_evaluator():

 
    # Filename -> datetime
  
    def filename_to_datetime(self, date_string):
        format_code = '%Y%m%d_%H%M%S'
        datetime_object = datetime.strptime(
            date_string[date_string.index('+1')+3:], format_code
        )
        return datetime_object
        
    def read_audio_file(self, file_name, audio_folder):
        audio_path = os.path.join(audio_folder, file_name)
        audio_amps, audio_sr = librosa.load(audio_path, sr=None)
        return audio_amps, audio_sr
        
    def get_annotation_information(self, audio_folder, annotation_folder, file_name):

        xmldoc = minidom.parse(os.path.join(annotation_folder, file_name + '.svl'))
        itemlist = xmldoc.getElementsByTagName('point')

        start_time, end_time, labels = [], [], []

        if len(itemlist) > 0:
            datetime_fromfile = self.filename_to_datetime(file_name)
            audio_amps, sr = self.read_audio_file(file_name + ".wav", audio_folder)

            for s in itemlist:
                label = str(s.attributes['label'].value)
                if label == '':
                    continue

                confidence = 10
                if ',' in label:
                    label, confidence = label.split(',')
                    confidence = int(confidence)

                if confidence != 10:
                    continue

                start_seconds = float(s.attributes['frame'].value) / sr
                duration_seconds = float(s.attributes['duration'].value) / sr

                start_dt = datetime_fromfile + timedelta(seconds=start_seconds)
                end_dt = start_dt + timedelta(seconds=duration_seconds)

                start_time.append(start_dt)
                end_time.append(end_dt)
                labels.append(label)

        return pd.DataFrame({
            'Start': start_time,
            'End': end_time,
            'Label': labels
        })
        
    def process_all_files_in_folder(self, audio_folder, annotation_folder, prediction_folder):

        df_all_annotations = pd.DataFrame()
        df_all_predictions = pd.DataFrame()

        for file in os.listdir(audio_folder):
            if file.endswith('.wav'):
                name = file[:-4]
                df = self.get_annotation_information(audio_folder, annotation_folder, name)
                df_all_annotations = pd.concat([df_all_annotations, df])

        for file in os.listdir(prediction_folder):
            if file.endswith('.svl'):
                name = file[:-4]
                df = self.get_annotation_information(audio_folder, prediction_folder, name)
                df_all_predictions = pd.concat([df_all_predictions, df])

        return df_all_annotations, df_all_predictions
        
    def compute_total_audio_hours(self, audio_folder):
        total_seconds = 0.0
        for f in os.listdir(audio_folder):
            if f.endswith(".wav"):
                y, sr = librosa.load(os.path.join(audio_folder, f), sr=None)
                total_seconds += len(y) / sr
        return total_seconds / 3600
        
    def count_metrics(
        self,
        df_true_annotations,
        df_pred_annotations,
        threshold_seconds,
        audio_folder,
        presence_label
    ):

        threshold = timedelta(seconds=threshold_seconds)

        df_true_pos = df_true_annotations[
            df_true_annotations['Label'] == presence_label
        ]

        TP = 0
        matched_predictions = set()

        # -------- TP & FN --------
        for _, true_row in df_true_pos.iterrows():
            overlaps = df_pred_annotations[
                (df_pred_annotations['Start'] <= true_row['End']) &
                (df_pred_annotations['End'] >= true_row['Start'])
            ]

            overlaps = overlaps[
                (overlaps['End'] - overlaps['Start']) >= threshold
            ]

            if not overlaps.empty:
                TP += 1
                matched_predictions.update(overlaps.index)

        FN = len(df_true_pos) - TP

        # -------- FP --------
        FP = len(df_pred_annotations) - len(matched_predictions)

        # -------- METRICS --------
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (
            2 * Precision * Recall / (Precision + Recall)
            if (Precision + Recall) > 0 else 0
        )

        # -------- FALSE ALARMS / HOUR --------
        total_audio_hours = self.compute_total_audio_hours(audio_folder)
        FA_per_hour = FP / total_audio_hours if total_audio_hours > 0 else np.nan

        # -------- DISPLAY --------
        print('=================================')
        print('++ EVENT-BASED DETECTION RESULTS ++')
        print(f'Target species           : {presence_label}')
        print(f'True presence calls      : {len(df_true_pos)}')
        print(f'True Positives (TP)      : {TP}')
        print(f'False Negatives (FN)     : {FN}')
        print(f'False Positives (FP)     : {FP}')
        print('---------------------------------')
        print(f'Precision                : {Precision:.3f}')
        print(f'Recall (TPR)             : {Recall:.3f}')
        print(f'F1-score                 : {F1:.3f}')
        print(f'False alarms / hour      : {FA_per_hour:.2f}')
        print('=================================')

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1,
            'FA_per_hour': FA_per_hour
        }
        
    def threshold_sweep(
        self,
        df_true,
        df_pred,
        audio_folder,
        presence_label,
        thresholds
    ):

        recalls, fa_rates = [], []

        for th in thresholds:
            res = self.count_metrics(
                df_true,
                df_pred,
                threshold_seconds=th,
                audio_folder=audio_folder,
                presence_label=presence_label
            )
            recalls.append(res['Recall'])
            fa_rates.append(res['FA_per_hour'])

        plt.figure()
        plt.plot(fa_rates, recalls, marker='o')
        plt.xlabel('False alarms / hour')
        plt.ylabel('Recall')
        plt.title('Detection Curve')
        plt.grid(True)
        plt.show()
