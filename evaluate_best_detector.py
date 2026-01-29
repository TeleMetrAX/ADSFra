import os
import numpy as np
import pandas as pd
import anomaly_detection_metrics_mod as anom
import json

# Algorithms to evaluate
algorithms = [
    'bayesChangePt', 'earthgeckoSkyline', 'windowedGaussian',
    'echoStateNetwork', 'relativeEntropy'
]

categories = ['Precision', 'Recall', 'F-score', 'MCC']
metric = 'F-score'

datasets_name = [
    'IOPS_3', 'IOPS_11', 'IOPS_19', 'IOPS_24', 'IOPS_46', 'IOPS_52', 'IOPS_59',
    'IOPS_68', 'IOPS_71', 'IOPS_81', 'IOPS_87', 'IOPS_89', 'IOPS_94', 'IOPS_96',
    'IOPS_102', 'IOPS_103', 'IOPS_104', 'IOPS_116', 'IOPS_118', 'IOPS_121',
    'IOPS_127', 'IOPS_128', 'IOPS_186', 'IOPS_192', 'IOPS_194', 'IOPS_199',
    'IOPS_218', 'IOPS_225', 'IOPS_236', 'IOPS_241', 'IOPS_242', 'IOPS_243',
    'IOPS_246', 'IOPS_255', 'IOPS_256', 'IOPS_261', 'IOPS_264', 'IOPS_267',
    'IOPS_269', 'IOPS_275', 'IOPS_285', 'IOPS_319', 'IOPS_327', 'IOPS_328',
    'IOPS_329', 'IOPS_330', 'IOPS_331', 'IOPS_333', 'IOPS_334', 'IOPS_337',
    'IOPS_338', 'IOPS_342', 'IOPS_344', 'IOPS_353', 'IOPS_365', 'IOPS_381',
    'IOPS_393', 'IOPS_394', 'IOPS_396', 'IOPS_401', 'IOPS_418', 'IOPS_427',
    'IOPS_450', 'IOPS_459', 'IOPS_461', 'IOPS_475', 'IOPS_478'
]


def evaluate_best_detectors(output_file, all_metrics_file):
    with open('thresholds.json') as f:
        thresholds = json.load(f)

    best_results = []
    all_metrics_data = []

    for dataset_name in datasets_name:
        print(f"\n-> Processing: {dataset_name}", flush=True)
        
        label_file = f'./data/labeled_{dataset_name}_labels.npy'
        if not os.path.exists(label_file):
             print(f"Warning: Label file not found for {dataset_name}. Skipping.")
             continue

        anomaly_labels = np.load(label_file)
        flag_indices = np.where(anomaly_labels == 1)[0]

        anomaly_scores = {}
        valid_results_found = False
        
        for algo in algorithms:
            result_path = f'./anomaly_detection_benchmarks/{algo}_{dataset_name}_results.csv'
            if os.path.exists(result_path):
                try:
                    df_res = pd.read_csv(result_path)
                    if 'anomaly_score' in df_res.columns:
                        anomaly_scores[algo] = df_res['anomaly_score'].values
                        valid_results_found = True
                    elif 'score' in df_res.columns:
                        anomaly_scores[algo] = df_res['score'].values
                        valid_results_found = True
                    elif 'Score' in df_res.columns:
                        anomaly_scores[algo] = df_res['Score'].values
                        valid_results_found = True
                    else:
                        anomaly_scores[algo] = np.zeros(len(flag_indices))
                except Exception as e:
                    print(f"Error loading {result_path}: {e}")
                    anomaly_scores[algo] = np.zeros(len(flag_indices))
            else:
                anomaly_scores[algo] = np.zeros(len(flag_indices))

        if not valid_results_found:
            print(f"Skipping {dataset_name}: No valid detector results found.")
            continue

        values = []
        for algo_name, scores in anomaly_scores.items():
            # print(f"  Calculating metrics for {algo_name}...")
            metrics = anom.calculate_anomaly_detection_metrics_main(scores, flag_indices, algo_name, 'thresholds.json')
            values.append([algo_name, *metrics])
            all_metrics_data.append([dataset_name, algo_name, metrics[0], metrics[1], metrics[2], metrics[3]])

        df_metrics = pd.DataFrame(values, columns=['Algorithm', *categories])
        
        # Find best algorithm based on metric
        if df_metrics.empty:
             print(f"Warning: No metrics calculated for {dataset_name}")
             continue
             
        best_algo_row = df_metrics.loc[df_metrics[metric].idxmax()]
        best_algo = best_algo_row['Algorithm']
        best_score = best_algo_row[metric]
        
        best_results.append([dataset_name, best_algo, best_algo_row['Precision'], best_algo_row['Recall'], best_algo_row['F-score'], best_algo_row['MCC']])
        
        print(f"Best algorithm for {dataset_name}: {best_algo} (F-score: {best_score:.4f})")

    # Save best results
    df_best = pd.DataFrame(best_results, columns=['Dataset', 'Best_Algorithm', 'Precision', 'Recall', 'F-score', 'MCC'])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_best.to_csv(output_file, index=False)
    print(f"\n[OK] Best algorithm results saved to: {output_file}")

    # Save all metrics
    df_all = pd.DataFrame(all_metrics_data, columns=['Dataset', 'Algorithm', 'Precision', 'Recall', 'F-score', 'MCC'])
    os.makedirs(os.path.dirname(all_metrics_file), exist_ok=True)
    df_all.to_csv(all_metrics_file, index=False)
    print(f"[OK] All metrics saved to: {all_metrics_file}")

if __name__ == "__main__":
    output_file = './test_results/best_detectors_per_dataset.csv'
    all_metrics_file = './test_results/all_detectors_metrics.csv'
    evaluate_best_detectors(output_file, all_metrics_file)
