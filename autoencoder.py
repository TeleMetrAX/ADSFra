import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import layers, Model
import random

# Set random seeds for reproducibility
SEED = 3
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Define input directories for each fingerprint type
fingerprint_sources = {
    'fingerprint2': './fingerprints/fingerprint_69features'
}

output_folder = './fingerprints/Autoencoder'
os.makedirs(output_folder, exist_ok=True)

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

n_components = 4  # Bottleneck size

def build_autoencoder(input_dim, bottleneck_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(input_layer)
    encoded = layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(encoded)
    bottleneck = layers.Dense(bottleneck_dim, activation='linear', name='bottleneck', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(encoded)

    # Decoder
    decoded = layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(bottleneck)
    decoded = layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(decoded)
    output_layer = layers.Dense(input_dim, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    return autoencoder, encoder

# Process each fingerprint type separately
for fp_name, input_folder in fingerprint_sources.items():
    print(f"\n=== Processing {fp_name} fingerprints ===")
    
    all_data_combined = pd.DataFrame()
    
    for dataset in datasets_name:
        # Construct file path based on fingerprint type naming convention
        if fp_name == 'fingerprint2':
            file_path = os.path.join(input_folder, f'labeled_{dataset}_values_fingerprint_69.csv')
        elif fp_name == 'PLif2':
            file_path = os.path.join(input_folder, f'{dataset}.csv')
        elif fp_name == 'TS2Vec':
            file_path = os.path.join(input_folder, f'{dataset}ts2vec_fingerprint.csv')
        
        if not os.path.exists(file_path):
            print(f"Warning: Missing {fp_name} for {dataset}")
            continue

        # Load data
        if fp_name == 'PLif2':
             data = pd.read_csv(file_path, header=None)
             data.columns = [f'plif_{i}' for i in range(data.shape[1])]
        else:
            data = pd.read_csv(file_path)

        if 'Timestamp' in data.columns:
            data_clean = data.drop(columns=['Timestamp'])
        else:
            data_clean = data

        data_clean.insert(0, 'Dataset', dataset)
        all_data_combined = pd.concat([all_data_combined, data_clean], ignore_index=True)

    if all_data_combined.empty:
        print(f"No data found for {fp_name}. Skipping.")
        continue

    # Standardize
    scaler = StandardScaler()
    numeric_cols = all_data_combined.select_dtypes(include=[np.number]).columns
    data_scaled = scaler.fit_transform(all_data_combined[numeric_cols])

    # Train Autoencoder
    print(f"Training Autoencoder for {fp_name}...")
    autoencoder, encoder = build_autoencoder(input_dim=data_scaled.shape[1], bottleneck_dim=n_components)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)
    print("Training complete.")
    
    # Generate reduced features
    bottleneck_components = encoder.predict(data_scaled)
    
    autoencoder_df = pd.DataFrame(bottleneck_components, columns=[f'AE_{i + 1}' for i in range(n_components)])
    autoencoder_df.insert(0, 'Dataset', all_data_combined['Dataset'])
    
    # Save results
    output_file_path = os.path.join(output_folder, f'{fp_name}_autoencoder_4.csv')
    autoencoder_df.to_csv(output_file_path, index=False)
    print(f"Saved reduced features to: {output_file_path}")

print("\nAll fingerprint types processed.")
