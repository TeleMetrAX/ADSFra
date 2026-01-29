import numpy as np
import pandas as pd
import os

def data_csv_to_npy(data_dir, save_to_dir):
    # Keresd meg az összes 'labeled_' előtaggal rendelkező fájlt
    for file_name in os.listdir(data_dir):
        if file_name.startswith('labeled_') and file_name.endswith('.csv'):
            csv_file = os.path.join(data_dir, file_name)

            # Létrehozott CSV fájl beolvasása
            df = pd.read_csv(csv_file)

            # Csak az értékek és anomáliák oszlopainak kiválasztása (első oszlop figyelmen kívül hagyva)
            values = df.iloc[:, 1].values  # Második oszlop (értékek)
            labels = df.iloc[:, 2].values  # Harmadik oszlop (anomália címkék)

            # .npy fájlok elmentése
            np.save(os.path.join(save_to_dir, file_name.replace('.csv', '_values.npy')), values)
            np.save(os.path.join(save_to_dir, file_name.replace('.csv', '_labels.npy')), labels)

            print(f"Értékek és címkék sikeresen elmentve .npy formátumban a {file_name}-hoz.")

def algorithm_result_csv_to_npy(data_path, save_to_dir):
    # Létrehozott CSV fájl beolvasása
    df = pd.read_csv(data_path)

    # Csak a negyedik oszlop (anomália címkék) kiválasztása
    predicted_anomalies = df.iloc[:, 3].values  # Negyedik oszlop (index 3)

    # Elmentett fájl nevének megadása
    output_file = os.path.join(save_to_dir, os.path.basename(data_path).replace('.csv', '_labels.npy'))

    # .npy fájl elmentése
    np.save(output_file, predicted_anomalies)

    print(f"Értékek és címkék sikeresen elmentve .npy formátumban a {output_file}-ba.")