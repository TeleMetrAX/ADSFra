import numpy as np
import pandas as pd
import os

def convert_out_to_csv(input_dir, output_dir):
    """
    Converts all .out files in the input directory to .csv files in the output directory.
    :param input_dir: Directory containing .out files.
    :param output_dir: Directory to save .csv files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('.out'):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace('.out', '.csv'))

            # Load .out file using pandas to handle comma-separated values
            try:
                data = pd.read_csv(input_path, header=None)  # Assuming no header in .out files
                data.to_csv(output_path, index=False, header=False)
                print(f"Converted {file} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {file}: {e}")

# Example usage
convert_out_to_csv('C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/YAHOO', 'C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/results/yahoocsv')