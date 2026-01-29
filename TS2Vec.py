import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ts2vec import TS2Vec
import time


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
    'Twitter_volume_CVS', 'Twitter_volume_FB', 'Twitter_volume_GOOG', 'Twitter_volume_IBM', 'Twitter_volume_KO',
    'Twitter_volume_PFE', 'Twitter_volume_UPS',

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

    # IOPS datasets
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

dataset_dict = {}  # Store processed datasets
all_series = []

# Define directories
data_dir = './data'
output_dir = './fingerprints/TS2Vec_4'
os.makedirs(output_dir, exist_ok=True)

for dataset_name in datasets_name:
    file_name = f"labeled_{dataset_name}_values.npy"
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        print(f"[!] File missing: {file_path}, skipping...")
        continue

    print(f"Loading: {dataset_name}")
    data = np.load(file_path).flatten()

    # Standardize data
    scaler = StandardScaler()

    data_standardized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Store dataset (reshape for TS2Vec: [1, sequence_length, 1])
    dataset_dict[dataset_name] = data_standardized.reshape(1, -1, 1)
    all_series.append(dataset_dict[dataset_name])

# Ensure all series have the same length for batch training
min_length = min(s.shape[1] for s in all_series)
all_series = [s[:, :min_length, :] for s in all_series]  # Truncate to min length

# Use all datasets for training
training_series = []
for name, series in dataset_dict.items():
    training_series.append(series[:, :min_length, :])

if not training_series:
    print("[!] No training datasets found. Check dataset names.")
    exit(1)

combined_data = np.concatenate(training_series, axis=0)  # Shape: (num_series, sequence_length, 1)
print(f"Training on {len(training_series)} datasets (NAB + Yahoo + IOPS).")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

model = TS2Vec(
    input_dims=1,
    device=device,
    output_dims=4  # Define output embedding size
)

# Check if model exists
model_save_dir = './ts2vec_models'
model_save_path = os.path.join(model_save_dir, 'ts2vec_model_4.pth')

if os.path.exists(model_save_path):
    print("\n=== Loading Existing Model ===")
    model.load(model_save_path)
    print(f"Model loaded from: {model_save_path}")
else:
    print("\n=== Aggregated Training Phase ===")
    n_epochs = 100

    start_time = time.time()
    loss_values = model.fit(combined_data, n_epochs=n_epochs, verbose=True)
    end_time = time.time()

    print("\nTraining Complete!")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Final Loss: {loss_values[-1]:.6f}")

    # Save the trained model
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")

print("\n=== Encoding Phase ===")

for dataset_name, data_np in dataset_dict.items():
    print(f"Encoding dataset: {dataset_name}")

    # Encode the dataset (get per-timestamp embeddings)
    timestamp_representations = model.encode(data_np)

    # Aggregate timestamps into a single fingerprint vector (e.g., by averaging)
    ts2vec_representation = np.mean(timestamp_representations, axis=1).flatten()

    # Save the representation as CSV
    output_file_path = os.path.join(output_dir, f"{dataset_name}ts2vec_fingerprint.csv")
    column_names = [f'ts2vec_{i}' for i in range(len(ts2vec_representation))]
    pd.DataFrame([ts2vec_representation], columns=column_names).to_csv(output_file_path, index=False)

    print(f"Saved fingerprint for {dataset_name} → {output_file_path}")

print("\nAll TS2Vec representations have been saved.")