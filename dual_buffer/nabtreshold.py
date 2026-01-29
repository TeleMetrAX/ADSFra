import os
import json
import numpy as np
import datetime
import sys
import os
import pandas as pd

# Add the parent directory of dual_buffer (which contains OnlineDetectors) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly_detectors.windowedGaussian.windowedGaussian_detector import WindowedGaussianDetector
from anomaly_detectors.relativeEntropy.relative_entropy_detector import RelativeEntropyDetector
from anomaly_detectors.knncad.knncad_detector import KnncadDetector
from anomaly_detectors.earthgeckoSkyline.earthgecko_skyline_detector import EarthgeckoSkylineDetector
from anomaly_detectors.contextOSE.context_ose_detector import ContextOSEDetector
from anomaly_detectors.bayesChangePt.bayes_changept_detector import BayesChangePtDetector
from anomaly_detectors.isolationForest.isolation_forest_detector import IsolationForestDetector
from anomaly_detectors.halfSpaceTree.half_space_tree_detector import HalfSpaceTreeDetector
from anomaly_detectors.echoStateNetwork.echo_state_network_detector import EchoStateNetworkDetector

# List of detectors to run (skip any not yet ready)
detectors = {
    "echoStateNetwork": EchoStateNetworkDetector,
}

# Profiles and their scoring functions
def fpr_fn_metrics(scores, labels, thr):
    preds = scores >= thr
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return tpr, fpr, fnr

profiles = {
    "standard": lambda tpr, fpr, fnr: (2 * tpr * (1 - fpr) / (tpr + 1 - fpr)) if (tpr + 1 - fpr) > 0 else 0.0,
    "reward_low_FP_rate": lambda tpr, fpr, fnr: tpr - fpr,
    "reward_low_FN_rate": lambda tpr, fpr, fnr: tpr - fnr,
}

# Load Yahoo dataset
def load_labeled_nab_csv(folder_path):
    data = {}
    for fname in os.listdir(folder_path):
        if not fname.endswith('.csv'):
            continue

        path = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(path)
            vals = df['value'].astype(float).values
            labels = df['label'].astype(int).values
            timestamps = np.arange(len(vals))  # vagy df['timestamp'] is lehetne

            data[fname] = {
                'timestamps': timestamps,
                'values': vals,
                'labels': labels
            }
        except Exception as e:
            print(f"⚠️ Hiba {fname}: {e}")
    return data



def run_detector(det_cls, data, probationary_period=750):
    class TSWrapper:
        def __init__(self, seconds):
            self.seconds = seconds
        def timestamp(self):
            return self.seconds

    scores_all, labels_all = [], []
    for ts_data in data.values():
        # most már .min() és .max() működik
        min_v = float(ts_data['values'].min())
        max_v = float(ts_data['values'].max())
        det = det_cls(input_min=min_v, input_max=max_v)
        det.probationaryPeriod = probationary_period
        if hasattr(det, 'initialize'):
            det.initialize()

        for t, v, l in zip(ts_data['timestamps'],
                            ts_data['values'],
                            ts_data['labels']):
            ts_obj = TSWrapper(t)
            out = det.handleRecord({"timestamp": ts_obj, "value": v})
            s = out[0] if isinstance(out, (list, tuple)) else out
            scores_all.append(float(s))
            labels_all.append(int(l))

    return np.array(scores_all), np.array(labels_all)


def optimize_threshold(scores, labels, n_steps=1001):
    thrs = np.linspace(0.0, 1.0, n_steps)
    best = {}
    for name, func in profiles.items():
        best_score, best_thr = -np.inf, 0.0
        for thr in thrs:
            tpr, fpr, fnr = fpr_fn_metrics(scores, labels, thr)
            score = func(tpr, fpr, fnr)
            if score > best_score:
                best_score, best_thr = score, thr
        best[name] = {"score": float(best_score), "threshold": float(best_thr)}
    return best


def main():
    yahoo_folder = "C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/labelled_NAB"  # adjust path as needed
    data = load_labeled_nab_csv(yahoo_folder)
    thresholds = {}

    for det_name, det_cls in detectors.items():
        print(f"Running {det_name}...")
        scores, labels = run_detector(det_cls, data)
        print(f"Optimizing thresholds for {det_name}...")
        thresholds[det_name] = optimize_threshold(scores, labels)

    # save to JSON
    with open('C:/Users/kristof/Documents/GitHub/onlab/kapott_kodok/Csabi/new_thresholds.json', 'w') as out:
        json.dump(thresholds, out, indent=4)
    print("thresholds.json generated.")

if __name__ == '__main__':
    main()
