import os
import pandas as pd
import glob

def convert_plif2(source_dir, output_file):
    print(f"Converting PLif2 from {source_dir}...")
    data = []
    files = glob.glob(os.path.join(source_dir, "*.csv"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        dataset_name = os.path.splitext(filename)[0]
        
        try:
            # PLif2: No header, single line
            df = pd.read_csv(file_path, header=None)
            if not df.empty:
                values = df.iloc[0].tolist()
                row = [dataset_name] + values
                data.append(row)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if data:
        # Determine max columns to name them AE_1...AE_n
        max_cols = max(len(row) - 1 for row in data)
        columns = ['Dataset'] + [f'AE_{i+1}' for i in range(max_cols)]
        
        out_df = pd.DataFrame(data, columns=columns)
        out_df.sort_values('Dataset', inplace=True)
        out_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(out_df)} rows.")
    else:
        print("No data found for PLif2.")

def convert_ts2vec(source_dir, output_file):
    print(f"Converting TS2Vec from {source_dir}...")
    data = []
    files = glob.glob(os.path.join(source_dir, "*ts2vec_fingerprint.csv"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Remove suffix for dataset name
        dataset_name = filename.replace("ts2vec_fingerprint.csv", "")
        
        try:
            # TS2Vec: Header exists, data on line 2 (index 0 after header)
            df = pd.read_csv(file_path)
            if not df.empty:
                values = df.iloc[0].tolist()
                row = [dataset_name] + values
                data.append(row)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if data:
        max_cols = max(len(row) - 1 for row in data)
        columns = ['Dataset'] + [f'AE_{i+1}' for i in range(max_cols)]
        
        out_df = pd.DataFrame(data, columns=columns)
        out_df.sort_values('Dataset', inplace=True)
        out_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(out_df)} rows.")
    else:
        print("No data found for TS2Vec.")

def convert_fingerprint2(source_dir, output_file):
    print(f"Converting fingerprint2 from {source_dir}...")
    data = []
    files = glob.glob(os.path.join(source_dir, "*_fingerprint_69.csv"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # Remove prefix and suffix
        dataset_name = filename.replace("labeled_", "").replace("_values_fingerprint_69.csv", "")
        
        try:
            # fingerprint2: Header exists, data on line 2
            df = pd.read_csv(file_path)
            if not df.empty:
                values = df.iloc[0].tolist()
                # Remove Dataset_Name column if present to avoid string in features
                if 'Dataset_Name' in df.columns:
                    # Assuming Dataset_Name is the first column, or we can drop it by name
                    # But values is a list, so we need index. 
                    # Safer to drop from df first? No, we already have values.
                    # Let's just check if first value is string and matches dataset_name?
                    # Or rely on column name.
                    if df.columns[0] == 'Dataset_Name':
                        values = values[1:]
                        
                row = [dataset_name] + values
                data.append(row)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if data:
        max_cols = max(len(row) - 1 for row in data)
        columns = ['Dataset'] + [f'AE_{i+1}' for i in range(max_cols)]
        
        out_df = pd.DataFrame(data, columns=columns)
        out_df.sort_values('Dataset', inplace=True)
        out_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} with {len(out_df)} rows.")
    else:
        print("No data found for fingerprint2.")

if __name__ == "__main__":
    base_dir = "./fingerprints"
    output_dir = os.path.join(base_dir, "Fingerprints")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_plif2(os.path.join(base_dir, "PLif2"), os.path.join(output_dir, "PLif2_results.csv"))
    convert_ts2vec(os.path.join(base_dir, "TS2Vec"), os.path.join(output_dir, "TS2Vec_results.csv"))
    
    # Convert TS2Vec variants
    for dim in [4, 8, 16, 32]:
        convert_ts2vec(
            os.path.join(base_dir, f"TS2Vec_{dim}"), 
            os.path.join(output_dir, f"TS2Vec_{dim}_results.csv")
        )

    convert_fingerprint2(os.path.join(base_dir, "fingerprint_69features"), os.path.join(output_dir, "fingerprint2_results.csv"))
