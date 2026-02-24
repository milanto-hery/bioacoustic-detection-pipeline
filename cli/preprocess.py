#!/usr/bin/env python3
"""
preprocess.py
CLI script for data preprocessing and feature extraction.
"""
import argparse
import os
import sys
import yaml

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bioacoustica.data.preprocessor import Preprocessor


def main():
    parser = argparse.ArgumentParser(description="BioAcoustica Preprocessing CLI")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--audio_dir", type=str, help="Path to raw audio files")
    parser.add_argument("--annotations_dir", type=str, help="Path to annotation files")
    parser.add_argument("--output_dir", type=str, help="Path to save processed data")
    parser.add_argument("--file_list", type=str, help="Text file containing list of files to process")
    parser.add_argument("--file_type", type=str, default="svl", choices=["svl", "raven"], help="Annotation file type")
    parser.add_argument("--audio_ext", type=str, help="Audio file extension")
    parser.add_argument("--bg_classes", type=str, nargs="*", help="Labels to treat as background/noise")
    parser.add_argument("--sample_rate", type=int, help="Target sample rate")
    parser.add_argument("--lowpass", type=int, help="Lowpass cutoff frequency")
    parser.add_argument("--duration", type=int, help="Segment duration in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    # Default parameters based on BioAcoustica standards or config
    params = {
        "audio_dir": args.audio_dir or "data/raw/audio",
        "annotations_dir": args.annotations_dir or "data/raw/annotations",
        "output_dir": args.output_dir or "data/processed",
        "file_list": args.file_list,
        "file_type": args.file_type,
        "audio_ext": args.audio_ext or ".wav",
        "bg_classes": args.bg_classes or ["noise"],
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
            
            # Species settings
            if "classes" in cfg:
                params["bg_classes"] = [cfg["classes"].get("absence", "noise")]
            
            # Audio settings
            if "audio" in cfg:
                a_cfg = cfg["audio"]
                params["sample_rate"] = a_cfg.get("sample_rate", params["sample_rate"])
                params["lowpass"] = a_cfg.get("lowpass_cutoff", params["lowpass"])
                params["duration"] = a_cfg.get("segment_duration", params["duration"])
                params["audio_ext"] = a_cfg.get("audio_extension", params["audio_ext"])
            
            # Spectrogram settings (stored in preprocessor defaults mostly, but we can pass them)
            if "spectrogram" in cfg:
                s_cfg = cfg["spectrogram"]
                params["n_fft"] = s_cfg.get("n_fft", 512)
                params["hop_length"] = s_cfg.get("hop_length", 128)
                params["n_mels"] = s_cfg.get("n_mels", 128)
                params["f_min"] = s_cfg.get("f_min", 0)
                params["f_max"] = s_cfg.get("f_max", 4000)

    # Overwrite with any explicitly provided CLI arguments
    if args.audio_dir: params["audio_dir"] = args.audio_dir
    if args.annotations_dir: params["annotations_dir"] = args.annotations_dir
    if args.output_dir: params["output_dir"] = args.output_dir
    if args.file_list: params["file_list"] = args.file_list
    if args.bg_classes: params["bg_classes"] = args.bg_classes

    # Validate required paths
    required = ["audio_dir", "annotations_dir", "output_dir", "file_list"]
    missing = [r for r in required if not params.get(r)]
    if missing:
        print(f"Error: Missing required arguments: {', '.join(missing)}")
        print("Please provide them via CLI or a --config file.")
        sys.exit(1)

    # Load file list
    if not os.path.exists(params["file_list"]):
        print(f"Error: File list not found: {params['file_list']}")
        sys.exit(1)
    
    with open(params["file_list"], "r") as f:
        files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    print(f"Initializing Preprocessor for {len(files)} files...")
    preprocessor = Preprocessor(
        audio_dir=params["audio_dir"],
        annotations_dir=params["annotations_dir"],
        output_dir=params["output_dir"],
        downsample_rate=params["sample_rate"],
        lowpass_cutoff=params["lowpass"],
        segment_duration=params["duration"],
        nyquist_rate=params["sample_rate"] // 2,
        n_fft=params.get("n_fft", 1024),
        hop_length=params.get("hop_length", 256),
        n_mels=params.get("n_mels", 128),
        f_min=params.get("f_min", 400),
        f_max=params.get("f_max", 4000)
    )

    X, Y = preprocessor.create_dataset_from_files(
        file_list=files,
        file_type=params["file_type"],
        audio_extension=params["audio_ext"],
        background_classes=params["bg_classes"],
        verbose=params["verbose"]
    )

    print(f"Dataset created. Total samples: {len(X)}")
    preprocessor.save_dataset(X, Y)
    print(f"Processed data saved to {params['output_dir']}")


if __name__ == "__main__":
    main()
