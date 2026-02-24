"""
app.py
BioAcoustica Web Dashboard — Streamlit interface for audio analysis, training, and evaluation.
"""
import os
import sys
import io
import tempfile
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import librosa
import librosa.display
import soundfile as sf

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bioacoustica.testing.predictor import Predictor
from bioacoustica.testing.evaluator import Evaluator
from bioacoustica.training.trainer import Trainer

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioAcoustica Dashboard",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #1a1a2e;}
    [data-testid="stSidebar"] * {color: #e0e0e0;}
    .main-title {font-size: 2.4rem; font-weight: 800; color: #4fc3f7; margin-bottom: 0;}
    .sub-title {font-size: 1rem; color: #90a4ae; margin-bottom: 1.5rem;}
    .metric-card {
        background: #1e2a38; border-radius: 12px; padding: 1.2rem;
        border-left: 4px solid #4fc3f7; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {font-weight: 600; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎧 BioAcoustica Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Species-Agnostic Bioacoustic Monitoring — Powered by Deep Learning</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Gibbon1.jpg/220px-Gibbon1.jpg", use_column_width=True)
    st.markdown("### ⚙️ Configuration")

    # Config loader
    CONFIG_PATH = os.path.join(ROOT, "configs", "gibbon.yaml")
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"species_name": "gibbon", "classes": {"positive": "gibbon", "absence": "no-gibbon", "order": ["no-gibbon", "gibbon"]},
                  "audio": {"sample_rate": 8000, "lowpass_cutoff": 3500, "segment_duration": 4},
                  "spectrogram": {"n_fft": 1024, "hop_length": 256, "n_mels": 128, "f_min": 400, "f_max": 4000},
                  "training": {"dropout_rate": 0.3, "fine_tune": False}}

    st.markdown("**📁 Data Upload**")
    uploaded_audio = st.file_uploader("Audio File (.wav)", type=["wav", "WAV"])
    uploaded_annotation = st.file_uploader("Annotation File (.svl)", type=["svl"])

    st.markdown("---")
    st.markdown("**🧠 Model Architecture**")
    arch_options = {"Custom CNN (Lightweight)": "custom_cnn",
                    "MobileNetV2 (Edge-Efficient)": "mobilenet_v2",
                    "EfficientNetB0 (Best Accuracy)": "efficientnet_b0"}
    arch_label = st.selectbox("Architecture", list(arch_options.keys()))
    arch = arch_options[arch_label]

    st.markdown("**⚗️ Hyperparameters**")
    epochs = st.slider("Epochs", 1, 100, config.get("training", {}).get("epochs", 20))
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.7, float(config.get("training", {}).get("dropout_rate", 0.3)), 0.05)
    fine_tune = st.toggle("Fine-tune Backbone", value=False, help="Unfreeze pre-trained layers (MobileNet/EfficientNet only)")

    st.markdown("---")
    st.markdown("**🎯 Detection Settings**")
    threshold = st.slider("Detection Threshold", 0.1, 0.99, 0.5, 0.05)
    model_files = [f for f in os.listdir(os.path.join(ROOT, "models")) if f.endswith(".h5")] if os.path.exists(os.path.join(ROOT, "models")) else []
    selected_model = st.selectbox("Trained Model", ["(none)"] + model_files)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Exploration", "🧠 Training", "📡 Inference", "📊 Evaluation"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: EXPLORATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("🔍 Audio Exploration")

    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format="audio/wav")

        # Load audio for analysis
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", f"{duration:.1f}s")
        col2.metric("Sample Rate", f"{sr:,} Hz")
        col3.metric("Samples", f"{len(y):,}")

        st.markdown("#### 🌊 Waveform")
        fig, ax = plt.subplots(figsize=(12, 2.5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4fc3f7", alpha=0.9)
        ax.set_xlabel("Time (s)", color="#90a4ae")
        ax.set_ylabel("Amplitude", color="#90a4ae")
        ax.tick_params(colors="#90a4ae")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 🎨 Log-Mel Spectrogram")
        sr_target = config["audio"]["sample_rate"]
        lowpass = config["audio"]["lowpass_cutoff"]

        y_ds = librosa.resample(y, orig_sr=sr, target_sr=sr_target) if sr != sr_target else y
        mel_spec = librosa.feature.melspectrogram(
            y=y_ds, sr=sr_target,
            n_fft=config["spectrogram"]["n_fft"],
            hop_length=config["spectrogram"]["hop_length"],
            n_mels=config["spectrogram"]["n_mels"],
            fmin=config["spectrogram"]["f_min"],
            fmax=config["spectrogram"]["f_max"]
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#0e1117")
        img = librosa.display.specshow(mel_db, sr=sr_target,
                                        hop_length=config["spectrogram"]["hop_length"],
                                        x_axis="time", y_axis="mel",
                                        fmin=config["spectrogram"]["f_min"],
                                        fmax=config["spectrogram"]["f_max"],
                                        ax=ax2, cmap="magma")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.tick_params(colors="#90a4ae")
        ax2.set_xlabel("Time (s)", color="#90a4ae")
        ax2.set_ylabel("Frequency (Hz)", color="#90a4ae")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3748")
        st.pyplot(fig2)
        plt.close(fig2)

        os.unlink(tmp_path)
    else:
        st.info("📂 Upload a `.wav` file in the sidebar to begin exploration.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("🧠 Model Training")

    data_dir = os.path.join(ROOT, "data", "processed")
    x_path = os.path.join(data_dir, "X.pkl")
    y_path = os.path.join(data_dir, "Y.pkl")

    col_info, col_action = st.columns([2, 1])
    with col_info:
        if os.path.exists(x_path) and os.path.exists(y_path):
            with open(x_path, "rb") as f:
                X_check = pickle.load(f)
            st.success(f"✅ Processed data found: **{len(X_check)}** samples · shape `{X_check.shape[1:]}`")
        else:
            st.warning("⚠️ No processed data found. Run `cli/preprocess.py` first.")

    with col_action:
        st.markdown("**Selected Configuration:**")
        st.code(f"arch: {arch}\nepochs: {epochs}\nbatch: {batch_size}\ndropout: {dropout_rate}\nfine_tune: {fine_tune}")

    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            st.error("No processed dataset found. Please preprocess audio first.")
        else:
            with open(x_path, "rb") as f:
                X = pickle.load(f)
            with open(y_path, "rb") as f:
                Y = pickle.load(f)

            class_order = config["classes"]["order"]
            models_dir = os.path.join(ROOT, "models")

            progress_bar = st.progress(0, text="Initializing training...")
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            log_placeholder = st.expander("📋 Training Log", expanded=False)

            class StreamlitCallback:
                """Keras-compatible callback that updates Streamlit UI."""
                def __init__(self, total_epochs):
                    self.total = total_epochs
                    self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    self.history["loss"].append(logs.get("loss", 0))
                    self.history["val_loss"].append(logs.get("val_loss", 0))
                    self.history["accuracy"].append(logs.get("accuracy", 0))
                    self.history["val_accuracy"].append(logs.get("val_accuracy", 0))

                    pct = int((epoch + 1) / self.total * 100)
                    progress_bar.progress(pct, text=f"Epoch {epoch+1}/{self.total} — val_loss: {logs.get('val_loss', 0):.4f}")

                    with log_placeholder:
                        st.text(f"Epoch {epoch+1:3d} | loss: {logs.get('loss', 0):.4f} | val_loss: {logs.get('val_loss', 0):.4f} | acc: {logs.get('accuracy', 0):.4f} | val_acc: {logs.get('val_accuracy', 0):.4f}")

                    # Live chart update
                    ep_list = list(range(1, epoch + 2))
                    fig_c, axes = plt.subplots(1, 2, figsize=(12, 3.5))
                    fig_c.patch.set_facecolor("#0e1117")
                    for a in axes:
                        a.set_facecolor("#0e1117")
                        a.tick_params(colors="#90a4ae")
                        for sp in a.spines.values():
                            sp.set_edgecolor("#2d3748")

                    axes[0].plot(ep_list, self.history["loss"], "#4fc3f7", label="Train")
                    axes[0].plot(ep_list, self.history["val_loss"], "#ef9a9a", label="Val", linestyle="--")
                    axes[0].set_title("Loss", color="#e0e0e0")
                    axes[0].set_xlabel("Epoch", color="#90a4ae")
                    axes[0].legend()

                    axes[1].plot(ep_list, self.history["accuracy"], "#81c784", label="Train")
                    axes[1].plot(ep_list, self.history["val_accuracy"], "#ffb74d", label="Val", linestyle="--")
                    axes[1].set_title("Accuracy", color="#e0e0e0")
                    axes[1].set_xlabel("Epoch", color="#90a4ae")
                    axes[1].legend()

                    chart_placeholder.pyplot(fig_c)
                    plt.close(fig_c)

            try:
                trainer = Trainer(output_dir=models_dir, seed=42)
                ui_callback = StreamlitCallback(epochs)

                # Patch trainer to inject our UI callback
                import tensorflow as tf

                X_train_tmp = X
                Y_train_tmp = Y

                # Prepare data through the data manager
                if X_train_tmp.ndim == 3:
                    X_3d = X_train_tmp
                else:
                    X_3d = X_train_tmp

                X_tr, X_val, y_tr, y_val = trainer.data_manager.prepare_data(
                    X_3d, Y_train_tmp, class_order, train_size=0.8, augment=True, verbose=False
                )
                if X_tr.ndim == 3:
                    X_tr = X_tr[..., np.newaxis]
                    X_val = X_val[..., np.newaxis]

                input_shape = X_tr.shape[1:]
                num_classes = len(class_order)

                trainer.compile_model(arch, input_shape, num_classes,
                                       dropout_rate=dropout_rate, fine_tune=fine_tune)

                from datetime import datetime
                from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

                class _EpochCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        ui_callback.on_epoch_end(epoch, logs)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{arch}_{ts}"
                best_path = os.path.join(models_dir, f"{model_name}_best.h5")
                final_path = os.path.join(models_dir, f"{model_name}_final.h5")

                callbacks = [
                    ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=0),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0),
                    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
                    _EpochCallback(),
                ]

                trainer.model.fit(
                    X_tr, y_tr,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0,
                )
                trainer.model.save(final_path)
                progress_bar.progress(100, text="✅ Training complete!")
                status_placeholder.success(f"Model saved: `{model_name}_best.h5`")
                st.session_state["last_model"] = best_path

            except Exception as e:
                st.error(f"Training failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: INFERENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.subheader("📡 Inference — Detect Calls")

    if selected_model == "(none)":
        st.info("🔧 Select a trained model in the sidebar.")
    elif uploaded_audio is None:
        st.info("📂 Upload an audio file in the sidebar.")
    else:
        model_path = os.path.join(ROOT, "models", selected_model)
        pred_dir = os.path.join(ROOT, "data", "predictions")
        audio_dir_tmp = tempfile.mkdtemp()

        # Save uploaded audio to a temp folder
        audio_bytes = uploaded_audio.getvalue()
        audio_name = uploaded_audio.name
        audio_tmp_path = os.path.join(audio_dir_tmp, audio_name)
        with open(audio_tmp_path, "wb") as f:
            f.write(audio_bytes)

        col_model, col_thresh = st.columns(2)
        col_model.info(f"**Model:** `{selected_model}`")
        col_thresh.info(f"**Threshold:** `{threshold}`")

        if st.button("🔍 Run Detection", use_container_width=True, type="primary"):
            os.makedirs(pred_dir, exist_ok=True)
            with st.spinner("Running inference..."):
                try:
                    predictor = Predictor(
                        audio_dir=audio_dir_tmp,
                        output_dir=pred_dir,
                        model_path=model_path,
                        downsample_rate=config["audio"]["sample_rate"],
                        lowpass_cutoff=config["audio"]["lowpass_cutoff"],
                        segment_duration=config["audio"].get("segment_duration", 4),
                        audio_extension=os.path.splitext(audio_name)[1],
                        nyquist_rate=config["audio"]["sample_rate"] // 2,
                        n_fft=config["spectrogram"]["n_fft"],
                        hop_length=config["spectrogram"]["hop_length"],
                        n_mels=config["spectrogram"]["n_mels"],
                        f_min=config["spectrogram"]["f_min"],
                        f_max=config["spectrogram"]["f_max"]
                    )
                    predictor.predict_file(audio_name, threshold=threshold)

                    # Read prediction SVL
                    base_name = os.path.splitext(audio_name)[0]
                    pred_svl = os.path.join(pred_dir, base_name + ".svl")

                    if os.path.exists(pred_svl):
                        evaluator = Evaluator(audio_dir=audio_dir_tmp)
                        df_pred = evaluator.read_svl(pred_svl, audio_name)

                        if df_pred.empty:
                            st.warning("No detections found above the threshold.")
                        else:
                            st.success(f"✅ Found **{len(df_pred)}** detection segments, merged to **{len(evaluator.merge_events_dt(df_pred))}** events.")

                            df_display = df_pred[["StartSec", "EndSec", "Label"]].copy()
                            df_display["Duration (s)"] = (df_display["EndSec"] - df_display["StartSec"]).round(2)
                            df_display = df_display.rename(columns={"StartSec": "Start (s)", "EndSec": "End (s)"})
                            df_display["Start (s)"] = df_display["Start (s)"].round(2)
                            df_display["End (s)"] = df_display["End (s)"].round(2)
                            st.dataframe(df_display, use_container_width=True)

                            st.session_state["last_predictions"] = df_pred
                            st.session_state["last_audio_name"] = audio_name
                            st.session_state["last_audio_dir"] = audio_dir_tmp
                    else:
                        st.warning("No predictions file generated. The model may have found no detections.")

                except Exception as e:
                    st.error(f"Inference error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.subheader("📊 Evaluation — Research-Grade Metrics")

    has_pred = "last_predictions" in st.session_state and not st.session_state["last_predictions"].empty
    has_annot = uploaded_annotation is not None

    if not has_pred:
        st.info("▶️ Run inference in the **Inference** tab to get predictions first.")
    elif not has_annot:
        st.info("📂 Upload a ground truth annotation `.svl` file in the sidebar.")
    else:
        if st.button("📊 Calculate Metrics", use_container_width=True, type="primary"):
            audio_dir_eval = st.session_state["last_audio_dir"]
            audio_name = st.session_state["last_audio_name"]

            with tempfile.NamedTemporaryFile(suffix=".svl", delete=False) as tmp_svl:
                tmp_svl.write(uploaded_annotation.read())
                gt_svl_path = tmp_svl.name

            with st.spinner("Evaluating..."):
                try:
                    evaluator = Evaluator(audio_dir=audio_dir_eval)
                    df_gt = evaluator.read_svl(gt_svl_path, audio_name)
                    df_pred = st.session_state["last_predictions"]

                    results = evaluator.evaluate(
                        df_gt=df_gt,
                        df_pred=df_pred,
                        target_label=config["classes"]["positive"],
                        min_overlap_pct=0.5,
                        test_files=[os.path.splitext(audio_name)[0]]
                    )

                    os.unlink(gt_svl_path)

                    st.markdown("### Results")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precision", f"{results['Precision']*100:.2f}%",
                               help="% of detections that matched a real call")
                    c2.metric("Recall", f"{results['Recall']*100:.2f}%",
                               help="% of real calls that were detected")
                    c3.metric("F1-Score", f"{results['F1']*100:.2f}%")
                    c4.metric("FA / Hr", f"{results['FP_per_hour']:.2f}",
                               help="False alarms per hour of audio")

                    st.markdown("---")
                    col_counts, col_chart = st.columns([1, 2])
                    with col_counts:
                        st.markdown("**Detection Summary**")
                        st.markdown(f"- Ground Truth Events: **{results['NumGT']}**")
                        st.markdown(f"- Merged Detections: **{results['NumDetections']}**")
                        st.markdown(f"- True Positives: **{results['TP']}**")
                        st.markdown(f"- False Positives: **{results['FP']}**")
                        st.markdown(f"- False Negatives: **{results['FN']}**")
                        st.markdown(f"- Analyzed Hours: **{results['TotalHours']:.2f} hrs**")

                    with col_chart:
                        fig_e, ax_e = plt.subplots(figsize=(6, 4))
                        fig_e.patch.set_facecolor("#0e1117")
                        ax_e.set_facecolor("#0e1117")
                        labels = ["True Positives", "False Positives", "False Negatives"]
                        values = [results["TP"], results["FP"], results["FN"]]
                        colors = ["#4fc3f7", "#ef9a9a", "#ffb74d"]
                        bars = ax_e.bar(labels, values, color=colors, edgecolor="#2d3748", width=0.5)
                        ax_e.tick_params(colors="#90a4ae")
                        ax_e.set_ylabel("Count", color="#90a4ae")
                        ax_e.set_title("Detection Breakdown", color="#e0e0e0", fontsize=13)
                        for bar, val in zip(bars, values):
                            ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                                       str(val), ha="center", va="bottom", color="#e0e0e0", fontweight="bold")
                        for sp in ax_e.spines.values():
                            sp.set_edgecolor("#2d3748")
                        st.pyplot(fig_e)
                        plt.close(fig_e)

                except Exception as e:
                    st.error(f"Evaluation error: {e}")
