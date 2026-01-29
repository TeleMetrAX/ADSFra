import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

plif = False
fp2 = True
ts2vec = False

# Define input folders
input_folder_fp2 = './fingerprints/fingerprint2'
input_folder_plif2 = './fingerprints/PLif2'
input_folder_ts2vec = './fingerprints/TS2Vec'
output_folder = './fingerprints/CombinedPCA'

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


    # IOPS datasets (every 10th)
    'IOPS_10', 'IOPS_20', 'IOPS_30', 'IOPS_40', 'IOPS_50', 'IOPS_60', 'IOPS_70', 'IOPS_80', 'IOPS_90', 'IOPS_100',
    'IOPS_110', 'IOPS_120', 'IOPS_130', 'IOPS_140', 'IOPS_150', 'IOPS_160', 'IOPS_170', 'IOPS_180', 'IOPS_190', 'IOPS_200',
    'IOPS_210', 'IOPS_220', 'IOPS_230', 'IOPS_240', 'IOPS_250', 'IOPS_260', 'IOPS_270', 'IOPS_280', 'IOPS_290', 'IOPS_300',
    'IOPS_310', 'IOPS_320', 'IOPS_330', 'IOPS_340', 'IOPS_350', 'IOPS_360', 'IOPS_370', 'IOPS_380', 'IOPS_390', 'IOPS_400',
    'IOPS_410', 'IOPS_420', 'IOPS_430', 'IOPS_440', 'IOPS_450', 'IOPS_460', 'IOPS_470', 'IOPS_480', 'IOPS_490', 'IOPS_500',
    'IOPS_510', 'IOPS_520', 'IOPS_530', 'IOPS_540', 'IOPS_550', 'IOPS_560', 'IOPS_570', 'IOPS_580'
]

use_kernel_pca = False  # Set to True if you want to use Kernel PCA, otherwise False
# number of principal components to keep
n_components = 69

plot = False
all_data_combined = pd.DataFrame()

for dataset in datasets_name:
    dfs_to_concat = []

    # 1. Load fingerprint2 data
    if fp2:
        fp2_path = os.path.join(input_folder_fp2, f'labeled_{dataset}_values_fingerprint.csv')
        if os.path.exists(fp2_path):
            df_fp2 = pd.read_csv(fp2_path)
            df_fp2.reset_index(drop=True, inplace=True)
            dfs_to_concat.append(df_fp2)
        else:
            print(f"Warning: Missing fingerprint2 for {dataset}")

    # 2. Load PLif2 data
    if plif:
        plif2_path = os.path.join(input_folder_plif2, f'{dataset}.csv')
        if os.path.exists(plif2_path):
            # PLif2 has no header, so we read it as header=None and assign names
            df_plif2 = pd.read_csv(plif2_path, header=None)
            df_plif2.columns = [f'plif_{i}' for i in range(df_plif2.shape[1])]
            df_plif2.reset_index(drop=True, inplace=True)
            dfs_to_concat.append(df_plif2)
        else:
            print(f"Warning: Missing PLif2 for {dataset}")

    # 3. Load TS2Vec data
    if ts2vec:
        ts2vec_path = os.path.join(input_folder_ts2vec, f'{dataset}ts2vec_fingerprint.csv')
        if os.path.exists(ts2vec_path):
            df_ts2vec = pd.read_csv(ts2vec_path)
            df_ts2vec.reset_index(drop=True, inplace=True)
            dfs_to_concat.append(df_ts2vec)
        else:
            print(f"Warning: Missing TS2Vec for {dataset}")

    if not dfs_to_concat:
        # No data found for this dataset with selected methods
        continue

    # Combine fingerprints (concatenate columns)
    data_clean = pd.concat(dfs_to_concat, axis=1)

    if 'Timestamp' in data_clean.columns:
        data_clean = data_clean.drop(columns=['Timestamp'])

    data_clean.insert(0, 'Dataset', dataset)
    all_data_combined = pd.concat([all_data_combined, data_clean], ignore_index=True)

scaler = StandardScaler()
if not all_data_combined.empty:
    numeric_cols = all_data_combined.select_dtypes(include=[np.number]).columns
    data_scaled = scaler.fit_transform(all_data_combined[numeric_cols])
else:
    print("Error: No data loaded. Check flags and file paths.")
    exit()

if use_kernel_pca:
    kernel_pca = KernelPCA(kernel='rbf', n_components=n_components)
    principal_components = kernel_pca.fit_transform(data_scaled)
else:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)

pca_df = pd.DataFrame(principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])
pca_df.insert(0, 'Dataset', all_data_combined['Dataset'])

os.makedirs(output_folder, exist_ok=True)
file_suffix = "kernel_pca" if use_kernel_pca else "pca"
combined_file_path = os.path.join(output_folder, f'combined_{file_suffix}_results.csv')
pca_df.to_csv(combined_file_path, index=False)
print(f"Combined PCA results saved to {combined_file_path}")