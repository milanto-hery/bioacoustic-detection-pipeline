#!/usr/bin/env python3
"""
evaluate.py
CLI script for evaluating detection performance against ground truth.
"""
import argparse
import os
import sys
import pandas as pd

import yaml

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bioacoustica.testing.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="BioAcoustica Evaluation CLI")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--audio_dir", type=str, help="Path to audio files")
    parser.add_argument("--gt_dir", type=str, help="Path to ground truth annotation files (.svl)")
    parser.add_argument("--pred_dir", type=str, help="Path to predicted annotation files (.svl)")
    parser.add_argument("--target_species", type=str, help="Target species label to evaluate")
    parser.add_argument("--overlap_pct", type=float, default=0.5, help="Minimum overlap percentage for TP")
    parser.add_argument("--file_list", type=str, help="Text file containing list of files to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")

    args = parser.parse_args()

    # Default parameters
    params = {
        "audio_dir": args.audio_dir or "data/raw/audio",
        "gt_dir": args.gt_dir or "data/raw/annotations",
        "pred_dir": args.pred_dir or "data/predictions",
        "target_species": args.target_species,
        "overlap_pct": args.overlap_pct,
        "file_list": args.file_list,
        "verbose": args.verbose
    }

    # Load from config if provided
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            params["target_species"] = cfg.get("species_name", params["target_species"])

    if not params["target_species"]:
        print("Error: Missing target species. Provide --target_species or a --config file.")
        sys.exit(1)

    evaluator = Evaluator(audio_dir=params["audio_dir"])

    print(f"Evaluating detections in {params['pred_dir']} against {params['gt_dir']}...")
    
    # Load files to evaluate
    if params["file_list"]:
        if not os.path.exists(params["file_list"]):
            print(f"Error: File list not found: {params['file_list']}")
            sys.exit(1)
        with open(params["file_list"], "r") as f:
            target_files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        target_files = [os.path.splitext(f)[0] for f in os.listdir(params["gt_dir"]) if f.endswith(".svl")]

    df_gt_list = []
    df_pred_list = []

    for base_name in target_files:
        if params["verbose"]:
            print(f"Evaluating: {base_name}")
            
        gt_path = os.path.join(params["gt_dir"], base_name + ".svl")
        pred_path = os.path.join(params["pred_dir"], base_name + ".svl")
        
        audio_file = base_name + ".wav"
        if not os.path.exists(os.path.join(params["audio_dir"], audio_file)):
            audio_file = base_name + ".WAV"
            if not os.path.exists(os.path.join(params["audio_dir"], audio_file)):
                 if params["verbose"]:
                    print(f"Warning: Audio file not found for {base_name}, using standard extension.")
                 audio_file = base_name + ".wav"

        # Ground Truth MUST exist
        if not os.path.exists(gt_path):
            print(f"Error: Ground truth file missing: {gt_path}")
            sys.exit(1)
            
        df_gt_list.append(evaluator.read_svl(gt_path, audio_file))

        # Prediction might not exist if 0 detections were found
        if os.path.exists(pred_path):
            df_pred_list.append(evaluator.read_svl(pred_path, audio_file))
        else:
            if params["verbose"]:
                print(f"  No predictions found for {base_name}, assuming 0 detections.")
            # Add empty DF but with AudioFile info for time tracking
            df_pred_list.append(pd.DataFrame(columns=["Start", "End", "Label", "StartSec", "EndSec", "AudioFile"]))

    if not df_gt_list:
        print("Error: No ground truth files found.")
        sys.exit(1)

    df_gt = pd.concat(df_gt_list)
    df_pred = pd.concat(df_pred_list) if df_pred_list else pd.DataFrame(columns=["Start", "End", "Label", "StartSec", "EndSec", "AudioFile"])

    results = evaluator.evaluate(
        df_gt=df_gt,
        df_pred=df_pred,
        target_label=params["target_species"],
        min_overlap_pct=params["overlap_pct"],
        test_files=target_files
    )

    print("\n" + "="*45)
    print(" EVALUATED FILES ")
    print("="*45)
    for f in target_files:
        print(f" - {f}")

    print("\n" + "="*45)
    print(" BIOACOUSTICA EVALUATION RESULTS ")
    print("="*45)
    print(f"Target Species:     {params['target_species']}")
    print(f"Ground Truth Events:{results['NumGT']}")
    print(f"Merged Detections:  {results['NumDetections']}")
    print(f"True Positives:     {results['TP']}")
    print(f"False Positives:    {results['FP']}")
    print(f"False Negatives:    {results['FN']}")
    print("-"*45)
    print(f"Precision:          {results['Precision']:.4f}")
    print(f"Recall:             {results['Recall']:.4f}")
    print(f"F1-Score:           {results['F1']:.4f}")
    print(f"False Alarms / Hr:  {results['FP_per_hour']:.2f}")
    print(f"Analyzed Hours:     {results['TotalHours']:.2f}")
    print("="*45)


if __name__ == "__main__":
    main()
