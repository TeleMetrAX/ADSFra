import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, skew, kurtosis


def compute_rff_features(X, D=100, gamma=0.1, random_state=None):
    """
    Compute Random Fourier Features (RFF) for approximating the RBF kernel.

    Parameters:
        X (numpy array): Data matrix of shape (n_samples, n_features).
        D (int): Number of random features.
        gamma (float): Parameter for the RBF kernel.
        random_state (int or None): Seed for reproducibility.

    Returns:
        Z (numpy array): Transformed data of shape (n_samples, D).
    """
    n_samples, n_features = X.shape
    rng = np.random.default_rng(random_state)
    # Draw random projection matrix from N(0, 2*gamma)
    W = rng.normal(loc=0, scale=np.sqrt(2 * gamma), size=(n_features, D))
    # Draw random bias uniformly from [0, 2*pi)
    b = rng.uniform(0, 2 * np.pi, size=(D,))

    # Compute the random Fourier features: Z = sqrt(2/D) * cos(X.dot(W) + b)
    Z = np.cos(np.dot(X, W) + b) * np.sqrt(2.0 / D)
    return Z


def learn_kernel_lds_rff(data, hidden_dim=5, D=100, gamma=0.1, random_state=None):
    """
    Learn a kernelized version of the LDS model using Random Fourier Features (RFF)
    to approximate the RBF kernel, then reduce the model to a lower dimension.

    Parameters:
        data (numpy array): 1-D time series data.
        hidden_dim (int): Number of time-delay snapshots and target reduced dimension.
        D (int): Number of random Fourier features (original dimension).
        gamma (float): RBF kernel parameter.
        random_state (int or None): Seed for reproducibility.

    Returns:
        A_reduced (numpy array): Reduced transition operator of shape (hidden_dim, hidden_dim).
        C_reduced (numpy array): Reduced output matrix of shape (n_samples, hidden_dim).
    """
    T = len(data)
    # Limit the hidden dimension based on available snapshots.
    hidden_dim = min(hidden_dim, max(2, T // 10))

    # Build time-delayed embedding: Each row is a snapshot (window) of the time series.
    X = np.array([data[i:T - hidden_dim + i] for i in range(hidden_dim)]).T  # shape: (num_snapshots, hidden_dim)

    # Compute RFF features for X.
    Z = compute_rff_features(X, D=D, gamma=gamma, random_state=random_state)  # shape: (n_samples, D)

    # Create shifted snapshots for Y (prediction one time tick ahead).
    Y = np.array([data[i + 1:T - hidden_dim + i + 1] for i in range(hidden_dim)]).T
    Z_Y = compute_rff_features(Y, D=D, gamma=gamma, random_state=random_state)

    # Estimate the transition operator in the RFF space using least squares.
    # Solve for A in Z_Y ≈ Z @ A.T  =>  A.T = pinv(Z) @ Z_Y, then take diagonal approximation.
    A_t = np.linalg.pinv(Z) @ Z_Y  # shape: (D, D)
    # For simplicity, approximate A by its diagonal elements.
    Lambda_approx = np.diag(np.diag(A_t))
    A_full = Lambda_approx  # shape: (D, D)

    # Reduce A to the target dimension: take the top-left hidden_dim x hidden_dim block.
    A_reduced = A_full[:hidden_dim, :hidden_dim]
    # Similarly, reduce C: use the first hidden_dim columns of Z.
    C_reduced = Z[:, :hidden_dim]

    return A_reduced, C_reduced


def compute_kernel_plif_features(A, C):
    """
    Compute the kernelized PLiF fingerprint features.

    Parameters:
        A (numpy array): Reduced transition matrix (hidden_dim x hidden_dim).
        C (numpy array): Reduced output matrix (n_samples x hidden_dim).

    Returns:
        fingerprint (list): Extracted fingerprint features.
    """
    # Step 1: Eigen decomposition of A.
    eigvals, V_eig = np.linalg.eig(A)  # V_eig will be (hidden_dim x hidden_dim)

    # Step 2: Compute the harmonic mixing matrix.
    Ch = C @ V_eig  # Now C (n_samples x hidden_dim) multiplied by V_eig (hidden_dim x hidden_dim)

    # Step 3: Remove phase information by taking element-wise magnitude.
    Cm = np.abs(Ch)

    # Step 4: Extract summary features.
    fingerprint = [
        np.mean(eigvals.real),  # Mean of real parts of eigenvalues
        np.var(eigvals.real),  # Variance
        skew(eigvals.real),  # Skewness
        kurtosis(eigvals.real),  # Kurtosis
        entropy(np.abs(eigvals.real) + 1e-10),  # Entropy of eigenvalues
        np.mean(Cm),  # Mean of the harmonic magnitude matrix
        np.var(Cm),  # Variance
        skew(Cm.flatten()),  # Skewness of flattened matrix
        kurtosis(Cm.flatten()),  # Kurtosis of flattened matrix
        entropy(np.abs(Cm.flatten()) + 1e-10)  # Entropy of flattened matrix
    ]

    return fingerprint


# Define dataset names (as before)
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

# Ensure the output directory exists
output_dir = './fingerprints/PLif2'
os.makedirs(output_dir, exist_ok=True)

# Main processing loop using the optimized kernelized PLiF with RFF and reduced dimensions
for dataset_name in datasets_name:
    file_name = "labeled_" + dataset_name + "_values"
    file_path = f'./data/{file_name}.npy'
    print(f"Loading file from: {file_path}")

    if not os.path.exists(file_path):
        print(f"[!] File missing: {file_path}, skipping...")
        continue

    data = np.load(file_path).flatten()

    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Learn the kernelized LDS model using RFF approximation.
    A_reduced, C_reduced = learn_kernel_lds_rff(data_standardized, hidden_dim=5, D=100, gamma=0.1, random_state=42)

    # Compute the kernelized PLiF fingerprints
    fingerprint_list = compute_kernel_plif_features(A_reduced, C_reduced)

    # Save the fingerprint vector to a CSV file
    output_file_path = os.path.join(output_dir, f"{dataset_name}.csv")
    with open(output_file_path, 'w') as f:
        f.write(','.join(map(str, fingerprint_list)))

    print(f"Fingerprints saved to {output_file_path}")