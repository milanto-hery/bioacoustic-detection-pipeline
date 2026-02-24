#!/usr/bin/env python3
"""
predict.py
CLI script for running inference on audio files using trained models.
"""
import argparse
import os
import sys
import yaml

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bioacoustica.testing.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="BioAcoustica Prediction CLI")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--audio_dir", type=str, help="Path to audio files for inference")
    parser.add_argument("--model_path", type=str, help="Path to trained model (.h5)")
    parser.add_argument("--output_dir", type=str, help="Path to save predictions (.svl)")
    parser.add_argument("--file_list", type=str, help="Text file containing list of files to process.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--audio_ext", type=str, help="Audio file extension")
    parser.add_argument("--sample_rate", type=int, help="Target sample rate")
    parser.add_argument("--lowpass", type=int, help="Lowpass cutoff frequency")
    parser.add_argument("--duration", type=int, help="Segment duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    # Default parameters
    params = {
        "audio_dir": args.audio_dir or "data/raw/audio",
        "model_path": args.model_path,
        "output_dir": args.output_dir or "data/predictions",
        "file_list": args.file_list,
        "threshold": args.threshold,
        "audio_ext": args.audio_ext or ".wav",
        "sample_rate": args.sample_rate or 8000,
        "lowpass": args.lowpass or 3500,
        "duration": args.duration or 4,
        "verbose": args.verbose
    }

    # Load from config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            
            # Audio settings
            if "audio" in cfg:
                a_cfg = cfg["audio"]
                params["sample_rate"] = a_cfg.get("sample_rate", params["sample_rate"])
                params["lowpass"] = a_cfg.get("lowpass_cutoff", params["lowpass"])
                params["duration"] = a_cfg.get("segment_duration", params["duration"])
                params["audio_ext"] = a_cfg.get("audio_extension", params["audio_ext"])
            
            # Spectrogram settings
            if "spectrogram" in cfg:
                s_cfg = cfg["spectrogram"]
                params["n_fft"] = s_cfg.get("n_fft", 512)
                params["hop_length"] = s_cfg.get("hop_length", 128)
                params["n_mels"] = s_cfg.get("n_mels", 128)
                params["f_min"] = s_cfg.get("f_min", 0)
                params["f_max"] = s_cfg.get("f_max", 4000)

    # Overwrite with any explicitly provided CLI arguments
    if args.audio_dir: params["audio_dir"] = args.audio_dir
    if args.model_path: params["model_path"] = args.model_path
    if args.output_dir: params["output_dir"] = args.output_dir

    # Validate required parameters
    required = ["audio_dir", "model_path", "output_dir"]
    missing = [r for r in required if not params.get(r)]
    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}")
        sys.exit(1)

    # Get file list
    if params["file_list"]:
        if not os.path.exists(params["file_list"]):
            print(f"Error: File list not found: {params['file_list']}")
            sys.exit(1)
        with open(params["file_list"], "r") as f:
            files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        files = [f for f in os.listdir(params["audio_dir"]) if f.endswith(params["audio_ext"])]

    print(f"Initializing Predictor for {len(files)} files...")
    predictor = Predictor(
        audio_dir=params["audio_dir"],
        output_dir=params["output_dir"],
        model_path=params["model_path"],
        downsample_rate=params["sample_rate"],
        lowpass_cutoff=params["lowpass"],
        segment_duration=params["duration"],
        audio_extension=params["audio_ext"],
        nyquist_rate=params["sample_rate"] // 2,
        n_fft=params.get("n_fft", 1024),
        hop_length=params.get("hop_length", 256),
        n_mels=params.get("n_mels", 128),
        f_min=params.get("f_min", 400),
        f_max=params.get("f_max", 4000)
    )

    for audio_file in files:
        if params["verbose"]:
            print(f"Predicting: {audio_file}")
        try:
            predictor.predict_file(audio_file, threshold=params["threshold"])
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print(f"Predictions saved to {params['output_dir']}")


if __name__ == "__main__":
    main()
