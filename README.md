# bioacoustic-detection-pipeline

This repository provides an **end-to-end pipeline for bioacoustic species call detection** using convolutional neural networks (CNNs).  
It is designed for **weakly supervised detection**, where **only presence calls are fully annotated**, and absence is implicit.


The pipeline is notebook-driven and focuses on:
- Audio preprocessing
- CNN training
- Batch prediction
- Event-based detection evaluation (PAM-oriented)


---


## 📁 Repository Structure



bioacoustic-detection-pipeline/
│
├── Audio
├── Annotations
├── src/ # Core reusable code
│ ├── data_process
│ ├── training
│ └── testing
│
├── notebooks/ # Demonstration notebooks (main entry point)
│ ├── training_demos.ipynb
│ └── testing_demos.ipynb
│
├── saved_models/
│ └── weights/ # Saved CNN weights (.keras)
│
├── saved_prediction/
│ ├── pred_1/ # 
│ └── pred_2/ # 
│
├── requirements.txt
├── .gitignore
└── README.md



---


## 🧠 Method Overview


1. Audio is segmented into short fixed-length windows  
2. Time–frequency features (e.g. spectrograms or log-mel) are extracted  
3. A CNN is trained to detect the presence of a target species  
4. The trained model is applied to long recordings  
5. Predictions are evaluated using **event-based metrics**, suitable for
   partially annotated bioacoustic datasets  


This approach is well suited for:
- Passive Acoustic Monitoring (PAM)
- Rare or low-data species
- Presence-only annotations


---


## 📓 Notebooks (How to Use This Repo)


### 1️⃣ Data Preparation  
**`01_data_preparation.ipynb`**


Purpose:
- Load WAV files and annotations
- Extract spectrogram features
- Build training labels
- Save processed datasets


Output:

data/processed/
├── X_S.pkl
├── Y_S.pkl


---


### 2️⃣ Model Training  
**`02_training_cnn.ipynb`**


Purpose:
- Load processed features
- Define CNN architecture
- Train the model
- Save trained weights


Output:

models/weights/
└── cnn_<species>_v1.keras



---


### 3️⃣ Batch Prediction  
**`03_prediction_batch.ipynb`**


Purpose:
- Load trained CNN weights
- Apply the model to multiple audio files
- Generate detection outputs


Output:

results/predictions/
├── file_001.svl
├── file_002.svl



---


### 4️⃣ Evaluation  
**`04_evaluation_metrics.ipynb`**


Purpose:
- Compare predictions with ground-truth annotations
- Compute **event-based detection metrics**
- Visualize detection performance


Metrics include:
- Precision
- Recall (True Positive Rate)
- F1-score
- False Alarms per Hour (FA/h)


Output:

results/
├── figures/
│ └── detection_curve.png
└── tables/
└── metrics_summary.csv



---


## 📊 Evaluation Philosophy


- Only presence calls are fully annotated  
- Absence is treated as implicit background  
- Metrics are computed at the **event level**, not frame level  
- No artificial true negatives are introduced  


This avoids biased accuracy scores and follows best practices in
bioacoustic detection research.


---


## ⚙️ Installation


Create a Python environment and install dependencies:


```bash
pip install -r requirements.txt

Tested with Python ≥ 3.9.
