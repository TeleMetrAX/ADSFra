import os
import numpy as np
import pandas as pd
import json
import glob
import anomaly_detection_metrics_mod as anom


# Configuration
RESULTS_DIR = "./anomaly_detection_benchmarks"
DATA_DIR = "./data"
THRESHOLDS_FILE = "thresholds.json"
OUTPUT_FILE = "./test_results/evaluation_summary.csv"

def load_thresholds(thresholds_file):
    with open(thresholds_file, 'r') as f:
        return json.load(f)

def evaluate_detectors():
    print("Starting evaluation...")
    
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory not found: {RESULTS_DIR}")
        return

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return

    thresholds = load_thresholds(THRESHOLDS_FILE)
    
    # Find all result CSV files
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*_results.csv"))
    print(f"Found {len(result_files)} result files.")

    evaluation_results = []

    for result_file in result_files:
        filename = os.path.basename(result_file)
        # Filename format: {detector}_{dataset}_results.csv
        # But dataset names can contain underscores (e.g. IOPS_1, Yahoo_A1real_1_data)
        # Detector names are known: bayesChangePt, earthgeckoSkyline, echoStateNetwork, windowedGaussian, isolationForest
        
        detector = None
        dataset = None
        
        # Heuristic to split filename
        known_detectors = [
            "bayesChangePt", "earthgeckoSkyline", "echoStateNetwork", 
            "windowedGaussian", "isolationForest", "relativeEntropy"
        ]
        
        for det in known_detectors:
            if filename.startswith(det + "_"):
                detector = det
                # Extract dataset name: remove detector_ and _results.csv
                dataset = filename[len(det)+1 : -len("_results.csv")]
                break
        
        if detector is None:
            print(f"Warning: Could not parse detector name from {filename}. Skipping.")
            continue

        print(f"Evaluating {detector} on {dataset}...")

        # Load scores
        try:
            df = pd.read_csv(result_file)
            if 'Score' not in df.columns:
                print(f"  Error: 'Score' column not found in {filename}. Skipping.")
                continue
            scores = df['Score'].values
        except Exception as e:
            print(f"  Error loading results: {e}. Skipping.")
            continue

        # Load ground truth labels
        # Try different label file naming conventions
        label_file = os.path.join(DATA_DIR, f"labeled_{dataset}_labels.npy")
        
        if not os.path.exists(label_file):
            # Try alternative naming if needed (e.g. for Yahoo)
            # Yahoo datasets in run_detectors_2.py were named like 'Yahoo_A1real_1_data'
            # Label file should be 'labeled_Yahoo_A1real_1_data_labels.npy' which matches the pattern.
            print(f"  Error: Label file not found: {label_file}. Skipping.")
            continue

        try:
            labels = np.load(label_file)
            # Ensure labels are 1D
            labels = labels.flatten()
        except Exception as e:
            print(f"  Error loading labels: {e}. Skipping.")
            continue

        # Check lengths
        if len(scores) != len(labels):
            print(f"  Warning: Length mismatch. Scores: {len(scores)}, Labels: {len(labels)}. Truncating to minimum.")
            min_len = min(len(scores), len(labels))
            scores = scores[:min_len]
            labels = labels[:min_len]

        # Get flag indices (indices where label is 1)
        flag_indices = np.where(labels == 1)[0]

        # Calculate metrics
        try:
            # anom.calculate_anomaly_detection_metrics_main expects:
            # anomaly_scores, flag_indices, detector, thresholds_file
            metrics = anom.calculate_anomaly_detection_metrics_main(
                scores, flag_indices, detector, THRESHOLDS_FILE
            )
            # Returns: [precision, recall, f_score, mcc_adj]
            
            precision, recall, f_score, mcc = metrics
            
            evaluation_results.append({
                'Detector': detector,
                'Dataset': dataset,
                'Precision': precision,
                'Recall': recall,
                'F-Score': f_score,
                'MCC': mcc
            })
            # print(f"  F-Score: {f_score:.4f}")

        except Exception as e:
            print(f"  Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    if evaluation_results:
        summary_df = pd.DataFrame(evaluation_results)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        summary_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nEvaluation complete. Summary saved to {OUTPUT_FILE}")
        
        # Print average F-Score per detector
        print("\nAverage F-Score per Detector:")
        print(summary_df.groupby('Detector')['F-Score'].mean())
    else:
        print("\nNo results evaluated.")

if __name__ == "__main__":
    evaluate_detectors()
