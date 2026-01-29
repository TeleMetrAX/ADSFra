import numpy as np
import pandas as pd
import os
import glob

# -------------
# CONFIG
# -------------
# Mappa, ahol a Yahoo .out fájlok vannak:
data_dir = 'C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/IOPS'
# Kimeneti mappa a CSV- és NPY-fájlok számára:
out_dir = 'C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/results/IOPS_results'
# -------------
# DETEKTOROK
# -------------
# Ide importáld a valós detector függvényeket:
# from my_detectors import Det1, Det2, ..., Det9
# A detektorfüggvényeknek fogadniuk kell egy 1D numpy-arrayt,
#és ugyanakkora hosszúságú numpy float-array-t visszaadniuk.
detectors = {
    'Det1': lambda x: np.zeros_like(x, dtype=float),
    'Det2': lambda x: np.zeros_like(x, dtype=float),
    'Det3': lambda x: np.zeros_like(x, dtype=float),
    'Det4': lambda x: np.zeros_like(x, dtype=float),
    'Det5': lambda x: np.zeros_like(x, dtype=float),
    'Det6': lambda x: np.zeros_like(x, dtype=float),
    'Det7': lambda x: np.zeros_like(x, dtype=float),
    'Det8': lambda x: np.zeros_like(x, dtype=float),
    'Det9': lambda x: np.zeros_like(x, dtype=float),
}

# -------------
# MAPPÁK LÉTREHOZÁSA
# -------------
os.makedirs(out_dir, exist_ok=True)

# -------------
# FÁJLOK ITERÁLÁSA
# -------------
# Keressük az összes *.out fájlt:
out_files = glob.glob(os.path.join(data_dir, '*.out'))

for filepath in out_files:
    # Alapnév a fájlból (pl. Yahoo_A1real_1)
    base = os.path.splitext(os.path.basename(filepath))[0].replace('_data', '')
    print(f"\n▶ Feldolgozás: {base}")

    # Beolvasás
    df = pd.read_csv(filepath, header=None, names=['value','label'])
    values = df['value'].values
    labels = df['label'].values.astype(int)

    # Labels mentése
    labels_path = os.path.join(out_dir, f'{base}_online_results_labels.npy')
    np.save(labels_path, labels)
    print(f"   • Labels mentve: {labels_path}")

    # Detektorok futtatása és CSV-kimenet
    for alg_name, alg_func in detectors.items():
        print(f"   - Detektor: {alg_name}")
        scores = alg_func(values)
        scores = np.array(scores, dtype=float)
        assert scores.shape[0] == values.shape[0], \
            f"[ERROR] {alg_name}: score-hossz nem egyezik a bemeneti hosszúsággal!"

        out_df = pd.DataFrame({
            'algorithm': alg_name,
            'dataset': base,
            'timestep': np.arange(len(values)),
            'anomaly_score': scores
        })
        csv_path = os.path.join(out_dir, f'{alg_name}_{base}_online_results.csv')
        out_df.to_csv(csv_path, index=False)
        print(f"     → CSV mentve: {csv_path}")

print("\n✅ Minden fájl feldolgozva.")
