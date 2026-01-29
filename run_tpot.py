
import numpy as np
import sklearn.metrics
import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import time
import os

# Configuration
SPLITS = ['NAB', 'YAHOO', 'IOPS']
BASE_OUTPUT_DIR = './tpot_dat_res/tpot_pipelines/'
GENERATIONS = 20
POPULATION_SIZE = 100
RANDOM_STATE = 42

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

def run_tpot_on_file(data_file_path, output_dir):
    print(f"\n=== Processing {data_file_path} ===")
    try:
        data = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Drop unnecessary columns
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    if 'Dataset' in data.columns:
        data = data.drop(columns=['Dataset'])
        
    if 'target' not in data.columns:
        print(f"Error: 'target' column missing in {data_file_path}")
        return None

    # Separate features and target
    X = data.drop(columns=['target'])
    y = data['target']
    

    # Drop non-numeric columns (e.g. accidental string columns)
    non_numeric = X.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"Dropping non-numeric columns: {list(non_numeric)}")
        X = X.drop(columns=non_numeric)

    # Clean data: Replace infinity with NaN and drop missing values
    initial_shape = X.shape
    
    # 1. Coerce to numeric (handles strings like 'inf' or garbled text)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    # 2. Replace explicit infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # 3. Check for values too large for float32 (TPOT/sklearn often cast to float32)
    # We use a much smaller threshold (1e10) than float32.max (3.4e38) to prevent 
    # overflow when TPOT applies transformations like PolynomialFeatures (squaring).
    # sqrt(3.4e38) is approx 1.8e19, so 1e10 is extremely safe for degree=2 interactions.
    SAFE_THRESHOLD = 1e10
    
    # Clip values to be within safe range
    X = X.clip(lower=-SAFE_THRESHOLD, upper=SAFE_THRESHOLD)

    # Combine X and y to drop NaNs row-wise
    data_combined = pd.concat([X, y], axis=1)
    data_combined = data_combined.dropna()
    
    if data_combined.shape[0] < initial_shape[0]:
        print(f"Dropped {initial_shape[0] - data_combined.shape[0]} rows containing NaN, Infinity, or non-numeric data.")
        
    X = data_combined.drop(columns=['target'])
    y = data_combined['target']

    if X.empty:
        print(f"Error: No data left after cleaning for {data_file_path}")
        return None

    # Split data 50-50
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=RANDOM_STATE, shuffle=True
    )

    # Initialize TPOT
    tpot = TPOTClassifier(
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        random_state=RANDOM_STATE,
        verbosity=2,
        n_jobs=8
    )

    # Fit TPOT
    start_time = time.time()
    tpot.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"TPOT fit finished in {duration:.2f} seconds")

    # Evaluate
    # Use sklearn accuracy_score directly to avoid TPOT scorer issues
    y_pred = tpot.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {score:.4f}")

    # Map labels to detector names
    algorithms = [
        'bayesChangePt', 'earthgeckoSkyline', 'windowedGaussian',
        'echoStateNetwork', 'relativeEntropy'
    ]
    # LabelEncoder sorts classes alphabetically
    sorted_algorithms = sorted(algorithms)
    label_map = {i: name for i, name in enumerate(sorted_algorithms)}

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred
    }, index=X_test.index)
    
    # Check if targets are encoded as numbers (indices) or are strings (names)
    # If they are numbers, map them back to names
    if pd.api.types.is_numeric_dtype(predictions_df['True_Label']):
         predictions_df['True_Detector'] = predictions_df['True_Label'].map(label_map)
         predictions_df['Predicted_Detector'] = predictions_df['Predicted_Label'].map(label_map)
    else:
         predictions_df['True_Detector'] = predictions_df['True_Label']
         predictions_df['Predicted_Detector'] = predictions_df['Predicted_Label']
    
    print("\n=== Predictions on Test Set ===")
    print(predictions_df[['True_Detector', 'Predicted_Detector']].to_string())
    
    # Save predictions
    filename = os.path.basename(data_file_path)
    predictions_filename = os.path.splitext(filename)[0] + '_predictions.csv'
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df.to_csv(predictions_path)
    print(f"\nPredictions saved to {predictions_path}")

    # Export pipeline
    pipeline_name = os.path.splitext(filename)[0] + '_pipeline.py'
    export_path = os.path.join(output_dir, pipeline_name)
    tpot.export(export_path)
    print(f"Pipeline exported to {export_path}")

    return {
        'File': filename,
        'Accuracy': score,
        'Duration': duration
    }

def main():
    for split_name in SPLITS:
        print(f"\n\n{'='*20} Processing Split: {split_name} {'='*20}")
        
        DATA_DIR = f'./tpot_dat_res/TPOT_data({split_name})/'
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, split_name)
        RESULTS_FILE = f'./tpot_dat_res/tpot_results_{split_name}.csv'
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        results = []
        
        if not os.path.exists(DATA_DIR):
            print(f"Data directory not found: {DATA_DIR}")
            continue
            
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                data_file_path = os.path.join(DATA_DIR, filename)
                
                result = run_tpot_on_file(data_file_path, OUTPUT_DIR)
                if result:
                    results.append(result)

        # Save results summary for this split
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_csv(RESULTS_FILE, index=False)
            print(f"\n[✓] Results summary saved to {RESULTS_FILE}")
        else:
            print(f"\n[!] No results to save for split {split_name}.")

if __name__ == "__main__":
    main()
