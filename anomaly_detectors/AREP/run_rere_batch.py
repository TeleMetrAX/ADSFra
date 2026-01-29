#!/usr/bin/env python3
import os
import sys
import pandas as pd
import importlib

import numpy as np

# NumPy 2.0 compat shim (used by older libraries)
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "PINF"):
    np.PINF = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf


# ---------------- set up sys.path to include project root ----------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'kapott_kodok')):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        break
    ROOT = parent

if not os.path.isdir(os.path.join(ROOT, 'kapott_kodok')):
    raise RuntimeError(
        f"Couldn't locate 'kapott_kodok' package relative to {__file__}. "
        f"Tried up to {ROOT}. Consider running with PYTHONPATH set to your repo root."
    )

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------------------------------

# --- Monkey-patch Sequential.reset_states to dispatch to RNN layers ---
import keras.models as _kmodels

def _seq_reset_states(self):
    for layer in self.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()
_kmodels.Sequential.reset_states = _seq_reset_states

# --- Patch standalone Keras LSTM to accept batch_input_shape and disable stateful ---
import keras.layers as _klayers
_orig_keras_LSTM = _klayers.LSTM
class _PatchedKerasLSTM(_orig_keras_LSTM):
    def __init__(self, *args, **kwargs):
        bshape = kwargs.pop('batch_input_shape', None)
        if bshape is not None:
            # Convert to input_shape and disable stateful
            kwargs['input_shape'] = bshape[1:]
            kwargs['stateful'] = False
        super().__init__(*args, **kwargs)
_klayers.LSTM = _PatchedKerasLSTM
# --- End keras.layers.LSTM patch ---

# --- Patch Lstm.train_lstm to reset states on any RNN layers ---
# Import the module where Lstm is defined
lstm_module = importlib.import_module('kapott_kodok.Burak.OnlineDetectors.AREP.ReRe.lstm_func')
_original_train = lstm_module.Lstm.train_lstm

def _patched_train(self, train_data, num_epochs, debug):
    x = train_data[:-1].reshape(1, train_data[:-1].shape[0], 1)
    y = train_data[-1].reshape(1)
    for _ in range(num_epochs):
        self.model.fit(x, y, batch_size=1, epochs=1, verbose=0, shuffle=False)
        # Reset states on any RNN layers
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        if debug:
            print('\t\tLSTM training done.')

# Apply patch
lstm_module.Lstm.train_lstm = _patched_train
# --- End train_lstm patch ---

# Dynamically locate project root containing 'kapott_kodok'
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'kapott_kodok')):
    PARENT = os.path.dirname(ROOT)
    if PARENT == ROOT:
        break
    ROOT = PARENT
sys.path.insert(0, ROOT)

# --- Patch auto_tune_offset to avoid division by zero ---
import importlib
# Load the auto_offset_comp module
_aoc = importlib.import_module('kapott_kodok.Burak.OnlineDetectors.AREP.ReRe.auto_offset_comp')
_orig_auto = _aoc.auto_tune_offset

def _patched_auto_tune_offset(self, time):
    try:
        _orig_auto(self, time)
    except ZeroDivisionError:
        if getattr(self, 'DEBUG', False):
            print(f"Skipping auto_tune_offset at timestep {time}: division by zero")
        # skip tuning when avg_dur_normal or denominator is zero
        return
# Apply the patch to the ReRe class method
import anomaly_detectors.AREP.ReRe.ReRe as _rere_mod
_rere_mod.ReRe.auto_tune_offset = _patched_auto_tune_offset

# Import the ReRe detector class after patches
from anomaly_detectors.AREP.ReRe.ReRe import ReRe


def process_file(detector, filename, input_dir, output_dir):
    """
    Runs the full ReRe pipeline on a single CSV file.
    Generates synthetic timestamps if none provided.
    Returns DataFrame with: algorithm, dataset, timestep, anomaly_score.
    """
    # Override load() to read value-only CSV and generate timestamps
    def _load_override(self):
        path = os.path.join(input_dir, filename)
        df = pd.read_csv(path, header=None, names=['value'])
        df['timestamp'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq='S')
        self.data = df
        self.length = len(df)
    detector.load = _load_override.__get__(detector, detector.__class__)

    # Execute pipeline
    detector.FILENAME = filename
    detector.param_refresh(filename)
    detector.load()
    detector.preprocess()
    detector.initialize_cons()
    detector.initialize_rere()
    detector.init_offset_compensation()
    detector.init_auto_offset_compensation()
    detector.init_auto_ws_ap()
    detector.init_timer()

    rows = []
    for t in range(detector.length):
        detector.next_timestep(t)
        detector.compensate_offset(t)
        detector.auto_tune_offset(t)
        detector.auto_tune_ws_ap(t)
        flag = detector.anomaly_aggr[t]
        rows.append([detector.__class__.__name__, filename, t, 1 if flag else 0])
    return pd.DataFrame(rows, columns=['algorithm', 'dataset', 'timestep', 'anomaly_score'])


def run_rere_batch(input_dir, output_dir, **detector_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not (fname.startswith('IOPS') and fname.endswith('.csv')):
            continue
        det = ReRe()
        det.rere_init_window = 30
        # Apply flag overrides
        for key, val in detector_kwargs.items():
            setattr(det, key, val)
        # Process and save
        df = process_file(det, fname, input_dir, output_dir)
        df = df[df['timestep'] >= det.rere_init_window]
        base = os.path.splitext(fname)[0]
        out_dir = os.path.join(output_dir, base)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'AREP_' + base + '_online_results.csv')
        df.to_csv(out_path, index=False)
        print(f"Processed {fname} -> {out_path}")

if __name__ == '__main__':
    from multiprocessing import Pool

    run_rere_batch(
        input_dir='../../data/iopscsv',
        output_dir='../../data/IOPS_results',
        USE_OFFSET_COMP=True,
        USE_AUTOMATIC_WS_AP=False,
        EVAL_EXPORT=False,
        TO_CSV=False,
        STATUS_BAR=True,
        NUM_EPOCHS=2,
        WINDOW_SIZE=30,
    )

    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    args = [(f, input_dir, output_dir, flags) for f in files]


    def worker(arg):
        fname, inp, out, kw = arg
        return process_file_batch(fname, inp, out, **kw)


    n_cores = multiprocessing.cpu_count() # or set to a specific number, uses all cores by default
    with Pool(processes=n_cores) as p:
        p.map(worker, args)

