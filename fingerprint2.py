import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis, entropy, mode, normaltest, iqr, moment
from scipy.signal import lombscargle, find_peaks
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. Feature Extraction Logic (The 69 Features)
# ==========================================

def calculate_hurst(X):
    """Estimate Hurst exponent using a simplified R/S analysis."""
    try:
        N = len(X)
        if N < 20: return 0.5
        min_N = 4
        rs_values = []
        # Split time series into chunks
        for n in [N, N//2, N//4, N//8]:
            if n < min_N: continue
            num_chunks = N // n
            for i in range(num_chunks):
                chunk = X[i*n : (i+1)*n]
                mean = np.mean(chunk)
                cum_dev = np.cumsum(chunk - mean)
                R = np.max(cum_dev) - np.min(cum_dev)
                S = np.std(chunk, ddof=1)
                if S == 0: continue
                rs_values.append(R/S)
        if not rs_values: return 0.5
        return np.mean(rs_values)
    except:
        return 0.5

def extract_features_69(values, timestamps=None):
    """
    Extracts the 69 features defined in arXiv:2307.13434v1.
    Categories: Statistical, Time, Distribution, Frequency, Behaviour.
    """
    # 1. Prepare Data
    x = np.array(values)
    n = len(x)
    
    # If sequence is too short, return zero vector
    if n < 5:
        return pd.DataFrame(np.zeros((1, 69)), columns=[f'feat_{i}' for i in range(69)])

    # Generate synthetic timestamps if none provided (0, 1, 2...)
    if timestamps is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.array(timestamps, dtype=float)

    # Relative times (start from 0)
    rt = t - t[0]
    
    # Time differences (spaces)
    dt = np.diff(t)
    if len(dt) == 0: dt = np.array([0.0])

    features = {}

    # ---------------------------------------------------------
    # A. Statistical-based Features (Table I)
    # ---------------------------------------------------------
    mean_val = np.mean(x)
    std_val = np.std(x)
    median_val = np.median(x)
    min_val = np.min(x)
    max_val = np.max(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr_val = q3 - q1
    
    # 1-4. Basic Stats
    features['Mean'] = mean_val
    features['Median'] = median_val
    features['Std'] = std_val
    features['Variance'] = np.var(x)
    
    # 5-6. Percent above/below mean
    features['Pct_Above_Mean'] = np.sum(x > mean_val) / n
    features['Pct_Below_Mean'] = np.sum(x < mean_val) / n
    
    # 7. Coefficient of Variation
    features['Coeff_Var'] = std_val / (mean_val + 1e-9)
    
    # 8. Kurtosis
    features['Kurtosis'] = kurtosis(x)
    
    # 9. Fisher-Pearson G1 Skewness
    features['Skew_Fisher'] = skew(x)
    
    # 10. Pearson SK1 Skewness: (Mean - Mode) / Std (using 3* for stability usually, but implementing standard definition)
    # Note: Scipy mode returns a mode object
    try:
        mode_res = mode(x, keepdims=True)
        mode_val = mode_res[0][0]
    except:
        mode_val = median_val
    features['Skew_Pearson_SK1'] = (mean_val - mode_val) / (std_val + 1e-9)
    
    # 11. Pearson SK2 Skewness: 3 * (Mean - Median) / Std
    features['Skew_Pearson_SK2'] = 3 * (mean_val - median_val) / (std_val + 1e-9)
    
    # 12. Galton Skewness: (Q1 + Q3 - 2*Median) / IQR
    features['Skew_Galton'] = (q1 + q3 - 2*median_val) / (iqr_val + 1e-9)
    
    # 13-17. Quartiles and Min/Max
    features['Q1'] = q1
    features['Q3'] = q3
    features['Min'] = min_val
    features['Max'] = max_val
    features['Min_Minus_Max'] = min_val - max_val
    
    # 18. Root Mean Square
    features['RMS'] = np.sqrt(np.mean(x**2))
    
    # 19. Average Dispersion (Mean Absolute Deviation)
    features['Avg_Dispersion'] = np.mean(np.abs(x - mean_val))
    
    # 20. Entropy
    hist_counts, _ = np.histogram(x, bins='doane', density=True)
    features['Entropy'] = entropy(hist_counts + 1e-9)
    
    # 21. Scaled Entropy
    features['Scaled_Entropy'] = features['Entropy'] / (np.log(len(hist_counts)) + 1e-9)
    
    # 22. Percent Deviation
    features['Pct_Deviation'] = std_val / (max_val - min_val + 1e-9)
    
    # 23. Mode
    features['Mode'] = mode_val
    
    # 24. Burstiness (Listed in Table I)
    features['Burstiness'] = std_val / (mean_val + 1e-9)

    # ---------------------------------------------------------
    # B. Time-based Features
    # ---------------------------------------------------------
    features['Time_Mean'] = np.mean(rt)
    features['Time_Median'] = np.median(rt)
    features['Time_Q1'] = np.percentile(rt, 25)
    features['Time_Q3'] = np.percentile(rt, 75)
    
    features['DiffTime_Mean'] = np.mean(dt)
    features['DiffTime_Median'] = np.median(dt)
    features['DiffTime_Min'] = np.min(dt)
    features['DiffTime_Max'] = np.max(dt)
    features['Duration'] = rt[-1]

    # ---------------------------------------------------------
    # C. Distribution-based Features
    # ---------------------------------------------------------
    # 1. Hurst Exponent
    features['Hurst_Exponent'] = calculate_hurst(x)
    
    # 2. Benford's Law (Compliance Check)
    # MSE of first digit distribution vs theoretical log10(1 + 1/d)
    first_digits = [int(str(abs(v)).lstrip('0.').replace('-','')[0]) for v in x if v != 0]
    if len(first_digits) > 5:
        counts = np.bincount(first_digits, minlength=10)[1:10]
        probs = counts / np.sum(counts)
        benford = np.log10(1 + 1/np.arange(1, 10))
        features['Benford_Law'] = np.mean((probs - benford)**2)
    else:
        features['Benford_Law'] = 0.0

    # 3. Normal Distribution (p-value)
    if n >= 8:
        _, p_val = normaltest(x)
        features['Normal_Dist'] = p_val
    else:
        features['Normal_Dist'] = 0.0
        
    # 4. Count Distribution (First half sum / Total sum)
    half = n // 2
    features['Count_Distribution'] = np.sum(x[:half]) / (np.sum(x) + 1e-9)
    
    # 5. Count Non-Zero Distribution
    nz = x[x != 0]
    if len(nz) > 0:
        nz_half = len(nz) // 2
        features['Count_NZ_Distribution'] = np.sum(nz[:nz_half]) / (np.sum(nz) + 1e-9)
    else:
        features['Count_NZ_Distribution'] = 0.0
        
    # 6. Time Distribution (Std of time differences)
    features['Time_Distribution'] = np.std(dt)
    
    # 7. Stationarity (Variance of first half vs second half)
    var1 = np.var(x[:half])
    var2 = np.var(x[half:])
    features['Stationarity'] = var1 / (var2 + 1e-9)

    # ---------------------------------------------------------
    # D. Frequency-based Features (Lomb-Scargle)
    # ---------------------------------------------------------
    # Defined for unevenly spaced time series
    freqs = np.linspace(0.01, 10, min(n, 500)) 
    pgram = lombscargle(t, x, freqs, normalize=True)
    
    # Power stats
    features['Power_Min'] = np.min(pgram)
    features['Power_Max'] = np.max(pgram)
    features['Power_Mean'] = np.mean(pgram)
    features['Power_Std'] = np.std(pgram)
    try:
        features['Power_Mode'] = mode(pgram, keepdims=True)[0][0]
    except:
        features['Power_Mode'] = 0
        
    # Frequencies of min/max power
    features['Freq_Max_Power'] = freqs[np.argmax(pgram)]
    features['Freq_Min_Power'] = freqs[np.argmin(pgram)]
    
    # Spectral properties
    total_energy = np.sum(pgram)
    norm_pgram = pgram / (total_energy + 1e-9)
    
    features['Spectral_Energy'] = total_energy
    features['Spectral_Entropy'] = entropy(norm_pgram + 1e-9)
    
    # Spectral Centroid
    centroid = np.sum(freqs * norm_pgram)
    features['Spectral_Centroid'] = centroid
    
    # Spectral Spread/Bandwidth
    spread = np.sqrt(np.sum(((freqs - centroid)**2) * norm_pgram))
    features['Spectral_Spread'] = spread
    
    # Spectral Skewness
    features['Spectral_Skewness'] = np.sum(((freqs - centroid)**3) * norm_pgram) / (spread**3 + 1e-9)
    
    # Spectral Kurtosis
    features['Spectral_Kurtosis'] = np.sum(((freqs - centroid)**4) * norm_pgram) / (spread**4 + 1e-9)
    
    # Spectral Flatness
    geo_mean = np.exp(np.mean(np.log(pgram + 1e-9)))
    features['Spectral_Flatness'] = geo_mean / (np.mean(pgram) + 1e-9)
    
    # Spectral Rolloff (85%)
    cum_energy = np.cumsum(norm_pgram)
    rolloff_idx = np.searchsorted(cum_energy, 0.85)
    features['Spectral_Rolloff'] = freqs[min(rolloff_idx, len(freqs)-1)]
    
    # Spectral Slope (Linear regression)
    try:
        lr = LinearRegression().fit(np.log(freqs).reshape(-1, 1), np.log(pgram + 1e-9))
        features['Spectral_Slope'] = lr.coef_[0]
    except:
        features['Spectral_Slope'] = 0.0
        
    # Spectral Flux (Rate of change)
    features['Spectral_Flux'] = np.mean(np.abs(np.diff(pgram)))
    
    # Spectral Periodicity (Significant peak existence)
    peaks, _ = find_peaks(pgram, height=np.mean(pgram) + 2*np.std(pgram))
    features['Periodicity_Spectral'] = 1.0 if len(peaks) > 0 else 0.0

    # ---------------------------------------------------------
    # E. Behavior-based Features
    # ---------------------------------------------------------
    # 1. Significant Spaces (Gaps > Mean + 2Std)
    thresh_space = np.mean(dt) + 2*np.std(dt)
    features['Sig_Spaces'] = np.sum(dt > thresh_space)
    
    # 2. Switching Ratio (Changes in value direction)
    diff_x = np.diff(x)
    switches = np.count_nonzero(np.diff(np.sign(diff_x)))
    features['Switching_Ratio'] = switches / (n - 1 + 1e-9)
    
    # 3. Transients (Values > Mean + 3Std)
    thresh_transient = mean_val + 3*std_val
    features['Transients'] = np.sum(x > thresh_transient)
    
    # 4. Count of Zeros
    features['Count_Zeros'] = np.sum(x == 0)
    
    # 5. Biggest Interval (Sum of values in window - defined as 1s in paper, here we use window=N/10)
    win_size = max(1, n // 10)
    features['Biggest_Interval'] = np.max(np.convolve(x, np.ones(win_size), mode='valid')) if n >= win_size else np.sum(x)
    
    # 6. Directions (Ratio of increasing steps)
    increasing = np.sum(diff_x > 0)
    features['Directions'] = increasing / (len(diff_x) + 1e-9)
    
    # 7. Periodicity (Time domain)
    # Simple autocorrelation check for lag > 0
    try:
        acf = np.correlate(x - mean_val, x - mean_val, mode='full')
        acf = acf[n:] # Right half
        peaks_acf, _ = find_peaks(acf)
        features['Periodicity_Time'] = acf[peaks_acf[0]] if len(peaks_acf) > 0 else 0
    except:
        features['Periodicity_Time'] = 0

    return pd.DataFrame([features])


# ==========================================
# 2. Dataset List
# ==========================================
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
    'Yahoo_A1real_1_data', 'Yahoo_A1real_2_data', 'Yahoo_A1real_3_data', 'Yahoo_A1real_4_data',
    'Yahoo_A1real_5_data', 'Yahoo_A1real_6_data', 'Yahoo_A1real_7_data', 'Yahoo_A1real_8_data',
    'Yahoo_A1real_9_data', 'Yahoo_A1real_10_data', 'Yahoo_A1real_11_data', 'Yahoo_A1real_12_data',
    'Yahoo_A1real_13_data', 'Yahoo_A1real_14_data', 'Yahoo_A1real_15_data', 'Yahoo_A1real_16_data',
    'Yahoo_A1real_17_data', 'Yahoo_A1real_18_data', 'Yahoo_A1real_19_data', 'Yahoo_A1real_20_data',
    'Yahoo_A1real_21_data', 'Yahoo_A1real_22_data', 'Yahoo_A1real_23_data', 'Yahoo_A1real_24_data',
    'Yahoo_A1real_25_data', 'Yahoo_A1real_26_data', 'Yahoo_A1real_27_data', 'Yahoo_A1real_28_data',
    'Yahoo_A1real_29_data', 'Yahoo_A1real_30_data', 'Yahoo_A1real_31_data', 'Yahoo_A1real_32_data',
    'Yahoo_A1real_33_data', 'Yahoo_A1real_34_data', 'Yahoo_A1real_35_data', 'Yahoo_A1real_36_data',
    'Yahoo_A1real_37_data', 'Yahoo_A1real_38_data', 'Yahoo_A1real_39_data', 'Yahoo_A1real_40_data',
    'Yahoo_A1real_41_data', 'Yahoo_A1real_42_data', 'Yahoo_A1real_43_data', 'Yahoo_A1real_44_data',
    'Yahoo_A1real_45_data', 'Yahoo_A1real_46_data', 'Yahoo_A1real_47_data', 'Yahoo_A1real_48_data',
    'Yahoo_A1real_49_data', 'Yahoo_A1real_50_data', 'Yahoo_A1real_51_data', 'Yahoo_A1real_52_data',
    'Yahoo_A1real_53_data', 'Yahoo_A1real_54_data', 'Yahoo_A1real_55_data', 'Yahoo_A1real_56_data',
    'Yahoo_A1real_57_data', 'Yahoo_A1real_58_data', 'Yahoo_A1real_59_data', 'Yahoo_A1real_60_data',
    'Yahoo_A1real_61_data', 'Yahoo_A1real_62_data', 'Yahoo_A1real_63_data', 'Yahoo_A1real_64_data',
    'Yahoo_A1real_65_data', 'Yahoo_A1real_66_data', 'Yahoo_A1real_67_data',
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

# ==========================================
# 3. Main Processing Loop
# ==========================================

# Configure paths
datasets_dir = './data'
output_dir = './fingerprints/fingerprint_69features'

os.makedirs(output_dir, exist_ok=True)

print(f"Starting extraction of 69 SFTS features for {len(datasets_name)} datasets...")

for dataset_name in datasets_name:
    # Construct file paths
    file_name = f"labeled_{dataset_name}_values"
    file_path = f"{datasets_dir}/{file_name}.npy"
    output_path = f"{output_dir}/{file_name}_fingerprint_69.csv"
    
    if os.path.exists(file_path):
        try:
            # Load Data
            data = np.load(file_path)
            
            # NOTE: We assume 'data' is the value sequence. 
            # We generate synthetic timestamps inside the function (0, 1, 2...).
            # If you have real timestamps, load them here and pass as second arg.
            df_features = extract_features_69(data)
            
            # Add dataset name column for reference
            df_features.insert(0, 'Dataset_Name', dataset_name)
            
            df_features.to_csv(output_path, index=False)
            print(f"[OK] Processed {dataset_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed processing {dataset_name}: {e}")
    else:
        print(f"[MISSING] File not found: {file_path}")

print("Batch processing complete.")