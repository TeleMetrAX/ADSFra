import os
import numpy as np
import pandas as pd
import anomaly_detection_metrics_mod as anom
import json
from sklearn.preprocessing import LabelEncoder

# ---- CHOOSE INPUT FILE ----
# input_file = './fingerprints/CombinedAutoencoder_LSH/autoencoder_lsh_bitvector.csv'
# input_file = './fingerprints/CombinedPCA/combined_pca_results.csv'
# output_file = './test_results/pca_with_best_algorithm.csv'

# Add HalfSpaceTree and EchoStateNetwork to the algorithms list
algorithms = [
    'bayesChangePt', 'earthgeckoSkyline', 'windowedGaussian',
    'echoStateNetwork', 'relativeEntropy'
]
draw_name = [
    'Bayesian Changepoint', 'Earthgecko Skyline', 'Windowed Gaussian',
    'Echo State Network', 'Relative Entropy'
]
categories = ['Precision', 'Recall', 'F-score', 'MCC']
metric = 'F-score'

datasets_name = [
    'ambient_temperature_system_failure', 'art_daily_flatmiddle', 'art_daily_jumpsdown',
    'art_daily_jumpsup', 'art_daily_nojump', 'art_increase_spike_density', 'art_load_balancer_spikes',
    'cpu_utilization_asg_misconfiguration', 'ec2_cpu_utilization_53ea38', 'ec2_cpu_utilization_5f5533',
    'ec2_cpu_utilization_24ae8d', 'ec2_cpu_utilization_77c1ca', 'ec2_cpu_utilization_825cc2',
    'ec2_cpu_utilization_ac20cd', 'ec2_cpu_utilization_fe7f93', 'ec2_disk_write_bytes_1ef3de',
    'ec2_disk_write_bytes_c0d644', 'ec2_network_in_257a54', 'ec2_request_latency_system_failure',
    'ec2_network_in_5abac7', 'elb_request_count_8c0756', 'exchange-2_cpc_results', 'exchange-2_cpm_results',
    'exchange-3_cpc_results', 'exchange-3_cpm_results', 'exchange-4_cpc_results', 'exchange-4_cpm_results',
    'grok_asg_anomaly', 'iio_us-east-1_i-a2eb1cd9_NetworkIn', 'machine_temperature_system_failure',
    'nyc_taxi', 'occupancy_6005', 'occupancy_t4013', 'rds_cpu_utilization_cc0c53', 'rds_cpu_utilization_e47b3b',
    'rogue_agent_key_hold', 'rogue_agent_key_updown', 'speed_6005', 'speed_7578', 'speed_t4013',
    'TravelTime_387', 'TravelTime_451', 'Twitter_volume_AAPL', 'Twitter_volume_AMZN', 'Twitter_volume_CRM',
    'Twitter_volume_CVS', 'Twitter_volume_FB', 'Twitter_volume_GOOG', 'Twitter_volume_IBM',
    'Twitter_volume_KO', 'Twitter_volume_PFE', 'Twitter_volume_UPS',

    # Yahoo datasets
    # A1real
    'Yahoo_A1real_1_data',
    'Yahoo_A1real_2_data',
    'Yahoo_A1real_3_data',
    'Yahoo_A1real_4_data',
    'Yahoo_A1real_5_data',
    'Yahoo_A1real_6_data',
    'Yahoo_A1real_7_data',
    'Yahoo_A1real_8_data',
    'Yahoo_A1real_9_data',
    'Yahoo_A1real_10_data',
    'Yahoo_A1real_11_data',
    'Yahoo_A1real_12_data',
    'Yahoo_A1real_13_data',
    'Yahoo_A1real_14_data',
    'Yahoo_A1real_15_data',
    'Yahoo_A1real_16_data',
    'Yahoo_A1real_17_data',
    'Yahoo_A1real_18_data',
    'Yahoo_A1real_19_data',
    'Yahoo_A1real_20_data',
    'Yahoo_A1real_21_data',
    'Yahoo_A1real_22_data',
    'Yahoo_A1real_23_data',
    'Yahoo_A1real_24_data',
    'Yahoo_A1real_25_data',
    'Yahoo_A1real_26_data',
    'Yahoo_A1real_27_data',
    'Yahoo_A1real_28_data',
    'Yahoo_A1real_29_data',
    'Yahoo_A1real_30_data',
    'Yahoo_A1real_31_data',
    'Yahoo_A1real_32_data',
    'Yahoo_A1real_33_data',
    'Yahoo_A1real_34_data',
    'Yahoo_A1real_35_data',
    'Yahoo_A1real_36_data',
    'Yahoo_A1real_37_data',
    'Yahoo_A1real_38_data',
    'Yahoo_A1real_39_data',
    'Yahoo_A1real_40_data',
    'Yahoo_A1real_41_data',
    'Yahoo_A1real_42_data',
    'Yahoo_A1real_43_data',
    'Yahoo_A1real_44_data',
    'Yahoo_A1real_45_data',
    'Yahoo_A1real_46_data',
    'Yahoo_A1real_47_data',
    'Yahoo_A1real_48_data',
    'Yahoo_A1real_49_data',
    'Yahoo_A1real_50_data',
    'Yahoo_A1real_51_data',
    'Yahoo_A1real_52_data',
    'Yahoo_A1real_53_data',
    'Yahoo_A1real_54_data',
    'Yahoo_A1real_55_data',
    'Yahoo_A1real_56_data',
    'Yahoo_A1real_57_data',
    'Yahoo_A1real_58_data',
    'Yahoo_A1real_59_data',
    'Yahoo_A1real_60_data',
    'Yahoo_A1real_61_data',
    'Yahoo_A1real_62_data',
    'Yahoo_A1real_63_data',
    'Yahoo_A1real_64_data',
    'Yahoo_A1real_65_data',
    'Yahoo_A1real_66_data',
    'Yahoo_A1real_67_data',

    # IOPS dataset
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


with open('thresholds.json') as f:
    thresholds = json.load(f)

label_encoder = LabelEncoder()
label_encoder.fit(algorithms)


def process_and_split_data(input_path, base_output_dir, metrics_output_path, metric='F-score'):
    if metric not in ['F-score', 'MCC']:
        raise ValueError("Invalid metric! Use 'F-score' or 'MCC'.")

    df_combined = pd.read_csv(input_path)
    print(f"Osszes rekord betoltve: {len(df_combined)}")
    
    if 'Dataset' not in df_combined.columns:
        print(f"Error: 'Dataset' column missing in {input_path}. Cannot align targets.")
        return

    best_algorithms = []
    valid_indices = []
    all_metrics_data = []

    for index, row in df_combined.iterrows():
        dataset_name = row['Dataset']
        print(f"\n-> Feldolgozas: {dataset_name}")
        
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
                        # print(f"Warning: No score column found in {result_path}")
                        anomaly_scores[algo] = np.zeros(len(flag_indices))
                except Exception as e:
                    print(f"Error loading {result_path}: {e}")
                    anomaly_scores[algo] = np.zeros(len(flag_indices))
            else:
                # print(f"Warning: Result file not found: {result_path}")
                anomaly_scores[algo] = np.zeros(len(flag_indices))

        if not valid_results_found:
            print(f"Skipping {dataset_name}: No valid detector results found.")
            continue

        values = []
        for algo_name, scores in anomaly_scores.items():
            # Only evaluate if we have non-zero scores or if we want to include failed ones as poor performers
            # But if all are zero, it's meaningless.
            # Assuming calculate_anomaly_detection_metrics_main handles zeros gracefully (likely low score)
            metrics = anom.calculate_anomaly_detection_metrics_main(scores, flag_indices, algo_name, 'thresholds.json')
            values.append([algo_name, *metrics])
            # metrics is [precision, recall, f_score, mcc_adj]
            all_metrics_data.append([dataset_name, algo_name, metrics[0], metrics[1], metrics[2], metrics[3]])

        df_metrics = pd.DataFrame(values, columns=['Algorithm', *categories])
        best_algo = df_metrics.loc[df_metrics[metric].idxmax(), 'Algorithm']
        
        # Check if the best score is actually 0 (meaning all failed)
        best_score = df_metrics[metric].max()
        if best_score == 0 and not valid_results_found:
             # This double check is redundant but safe
             print(f"Skipping {dataset_name}: Best score is 0 (likely missing results).")
             continue

        best_algorithms.append(best_algo)
        valid_indices.append(index)
        
        # Get metrics for the best algorithm for printing
        best_metrics = df_metrics.loc[df_metrics['Algorithm'] == best_algo].iloc[0]
        print(f"Legjobb algoritmus: {best_algo} (Precision: {best_metrics['Precision']:.4f}, Recall: {best_metrics['Recall']:.4f}, F-score: {best_metrics['F-score']:.4f}, MCC: {best_metrics['MCC']:.4f})")

    # Filter df_combined to keep only valid rows
    df_combined = df_combined.loc[valid_indices].copy()
    
    # Remove Dataset column for training if desired, or keep it. 
    # The original code dropped it. Let's drop it but maybe keep it for debugging? 
    # The user's prompt implies they want to use this for training, so dropping is safer.
    # df_combined = df_combined.drop(columns=['Dataset'])
    # Célváltozó létrehozása
    encoded = label_encoder.transform(best_algorithms)
    df_combined['target'] = encoded

    # Define Splits
    splits = {
        'NAB': lambda x: not (str(x).startswith('IOPS_') or str(x).startswith('Yahoo_')), # Test = NAB
        'IOPS': lambda x: str(x).startswith('IOPS_'), # Test = IOPS
        'YAHOO': lambda x: str(x).startswith('Yahoo_'), # Test = Yahoo
        'RANDOM': 'random' # Special case
    }

    unique_datasets = df_combined['Dataset'].unique()
    
    for split_name, condition in splits.items():
        print(f"  Generating split: {split_name}")
        
        if split_name == 'RANDOM':
            if len(unique_datasets) < 62:
                print(f"Warning: Not enough datasets ({len(unique_datasets)}) to select 62 for testing. Using 20% instead.")
                test_datasets = np.random.choice(unique_datasets, int(len(unique_datasets) * 0.2), replace=False)
            else:
                # Set seed for reproducibility
                np.random.seed(42)
                test_datasets = np.random.choice(unique_datasets, 62, replace=False)
            mask_test = df_combined['Dataset'].isin(test_datasets)
        else:
            mask_test = df_combined['Dataset'].apply(condition)
            
        df_test = df_combined[mask_test]
        df_train = df_combined[~mask_test]
        
        # Define output paths
        train_dir = os.path.join(base_output_dir, f'TPOT_train_data({split_name})')
        test_dir = os.path.join(base_output_dir, f'TPOT_test_data({split_name})')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_output_path = os.path.join(train_dir, os.path.basename(input_path).replace('.csv', '_best_algorithm.csv'))
        test_output_path = os.path.join(test_dir, os.path.basename(input_path).replace('.csv', '_best_algorithm.csv'))

        df_train.to_csv(train_output_path, index=False)
        df_test.to_csv(test_output_path, index=False)
        print(f"    Saved {split_name} split: Train={len(df_train)}, Test={len(df_test)}")

    # Save all metrics to CSV
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    df_all_metrics = pd.DataFrame(all_metrics_data, columns=['Dataset', 'Algorithm', 'Precision', 'Recall', 'F-score', 'MCC'])
    df_all_metrics.to_csv(metrics_output_path, index=False)
    print(f"[OK] Minden metrika mentve ide: {metrics_output_path}")


# ---- Futtatás ----
# Define the input directory containing fingerprint files
input_dir = './fingerprints/Fingerprints'
base_output_dir = './tpot_dat_res' # Base dir for TPOT_train_data(...) folders
test_results_dir = './test_results'
os.makedirs(test_results_dir, exist_ok=True)

# Iterate over all .csv files in the input directory
if os.path.exists(input_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            print(f"\n=== Evaluating {filename} ===")
            
            # Construct metrics output filename
            metrics_output_filename = f"{os.path.splitext(filename)[0]}_all_metrics.csv"
            metrics_output_path = os.path.join(test_results_dir, metrics_output_filename)
            
            process_and_split_data(input_path, base_output_dir, metrics_output_path, metric=metric)
else:
    print(f"Error: Input directory not found: {input_dir}")
