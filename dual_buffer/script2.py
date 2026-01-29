import os
import json
import numpy as np
import sys

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
    "earthgeckoSkyline": EarthgeckoSkylineDetector,
    "contextOSE": ContextOSEDetector,
    "bayesChangePt": BayesChangePtDetector,
    "windowedGaussian": WindowedGaussianDetector,
    "relativeEntropy": RelativeEntropyDetector,
    "knncad": KnncadDetector,
    "isolationForest": IsolationForestDetector,
    "halfSpaceTree": HalfSpaceTreeDetector,
    "echoStateNetwork": EchoStateNetworkDetector,
}

profiles = {
    "standard": lambda tpr, fpr, fnr: (2 * tpr * (1 - fpr) / (tpr + 1 - fpr)) if (tpr + 1 - fpr) > 0 else 0.0,
    "reward_low_FP_rate": lambda tpr, fpr, fnr: tpr - fpr,
    "reward_low_FN_rate": lambda tpr, fpr, fnr: tpr - fnr,
}

def optimize_threshold(scores, labels, n_steps=1001):
    """Exactly as in script.py: search 0→1 for best score／threshold."""
    thrs = np.linspace(0.0, 1.0, n_steps)
    best = {}
    for name, func in profiles.items():
        best_score, best_thr = -np.inf, 0.0
        for thr in thrs:
            tpr = np.sum((scores >= thr) & (labels == 1)) / max(1, np.sum(labels == 1))
            fpr = np.sum((scores >= thr) & (labels == 0)) / max(1, np.sum(labels == 0))
            fnr = np.sum((scores <  thr) & (labels == 1)) / max(1, np.sum(labels == 1))
            score = func(tpr, fpr, fnr)
            if score > best_score:
                best_score, best_thr = score, thr
        best[name] = {"score": float(best_score), "threshold": float(best_thr)}
    return best

def load_yahoo_csv(folder_path):
    """
    Reads all Yahoo_A1real*.csv files in folder_path.
    Expects two comma-separated columns: value,label
    Returns dict fname → {'values': np.array, 'labels': np.array}.
    """
    data = {}
    for fname in os.listdir(folder_path):
        if not (fname.startswith('Yahoo_A1real') and fname.endswith('.csv')):
            continue
        vs, ls = [], []
        with open(os.path.join(folder_path, fname), 'r') as f:
            for line in f:
                v, l = line.strip().split(',')
                vs.append(float(v));  ls.append(int(l))
        data[fname] = {
            'values': np.array(vs),
            'labels': np.array(ls, dtype=int)
        }
    return data

def run_detector(det_cls, data, probationary_period=750):
    """
    Mirrors script.py’s run_detector, but on our CSV data.
    Collects scores & labels across *all* series.
    """
    class DummyTS:
        def __init__(self, t): self.t = t
        def timestamp(self): return self.t

    scores, labels = [], []
    for series in data.values():
        det = det_cls(min(series['values']), max(series['values']))
        det.probationaryPeriod = probationary_period
        if hasattr(det, 'initialize'):
            det.initialize()

        for t, (v, l) in enumerate(zip(series['values'], series['labels'])):
            out = det.handleRecord({"timestamp": DummyTS(t), "value": v})
            s = out[0] if isinstance(out, (list, tuple)) else out
            scores.append(s);  labels.append(l)

    return np.array(scores), np.array(labels)

def main():
    yahoo_folder = r'C:\Users\kristof\Documents\GitHub\onlab\kapott_kodok\Csabi\results\yahoocsv'
    data = load_yahoo_csv(yahoo_folder)

    thresholds = {}
    for det_name, det_cls in detectors.items():
        print(f"Optimizing thresholds for {det_name}…")
        scores, labels = run_detector(det_cls, data)
        thresholds[det_name] = optimize_threshold(scores, labels)

    out_path = os.path.join(os.path.dirname(yahoo_folder), 'yahoo_thresholds.json')
    with open(out_path, 'w') as f:
        json.dump(thresholds, f, indent=4)
    print(f"Written thresholds → {out_path}")

if __name__ == '__main__':
    main()