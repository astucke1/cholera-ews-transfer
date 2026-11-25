####################################################################
# apply_cholera_trained.py -- apply the trained classifiers to the held-out SEIARB set (pipeline step 6)
####################################################################

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import re
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import tensorflow as tf
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

import keras
from keras.models import load_model


SEED = 1988
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
try:
    gpus = tf.config.list_physical_devices("GPU")
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

# ---- path convention ----
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent

# SEIIRB simulation outputs
# Which SEIARB label rule this run evaluates: "ramp" (slow-parameter rule; main results) or
# "index" (seasonal-index rule, sensitivity analysis). Inputs come from
# a_simulations/3_seiirb/output/SEIIRB_<rule>_rule and every output folder below carries the same suffix,
# so the two rules dont overwrite each other. Run the script once per rule.
_parser = argparse.ArgumentParser(description="Apply the cholera-trained classifiers to the SEIARB residual series.")
_parser.add_argument("--label-rule", choices=("ramp", "index"), default="ramp",
                     help="SEIARB transition label rule (default: ramp, the rule behind the main results)")
SEIIRB_LABEL_RULE = _parser.parse_args().label_rule
SEIIRB_OUTPUT = REPO_ROOT / "a_simulations" / "3_seiirb" / "output" / f"SEIIRB_{SEIIRB_LABEL_RULE}_rule"
SIM_RESID_DIR = SEIIRB_OUTPUT / "output_resids"
LABELS_DIR    = SEIIRB_OUTPUT / "output_labels"


MODEL_DIRS = [
    REPO_ROOT / "b_training" / "train100",
    REPO_ROOT / "b_training" / "train500",
]


OUT_BASE = SCRIPT_DIR / "output" / f"predictions_{SEIIRB_LABEL_RULE}_rule"


TCAP_IF_MISSING = True


RESIDUAL_COL_PRIMARY  = "residuals"
RESIDUAL_COL_FALLBACK = "residual"


MULT_FACTOR = 10

LENGTH_TAG_TO_LEN = {
    "length100": 100,
    "length500": 500,
}

# model grids
CHAK_MODELS = [(m, kk) for m in (1, 2) for kk in (1, 2)]
BURY_MODELS = [(x, y) for x in range(1, 11) for y in (1, 2)]


#  toggles
SAMPLE_N       = None       # set to e.g. 10 for smoke test, None for all
RUN_ONLY       = "chak"     # "chak" | "bury" | "both"
LENGTHS        = ["length100", "length500"]
WRITE_PER_MODEL = True
OVERWRITE       = False
SKIP_IF_EXISTS  = True


def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_residual_csvs(dir_path):
    return sorted(dir_path.glob("resids_batch*_sim*.csv"), key=lambda p: _natural_sort_key(p.name))

SIM_TAG_RE = re.compile(r"resids_batch(\d+)_sim(\d+)$")

def parse_sim_tag(stem):
    m = SIM_TAG_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognized residual filename stem: {stem}")
    return int(m.group(1)), int(m.group(2))

def load_t_outbreak(labels_dir, batch, sim_index):
    f = labels_dir / f"outbreak_times_batch{batch}.csv"
    if not f.exists():
        if TCAP_IF_MISSING:
            return float("inf")
        raise FileNotFoundError(f"Missing: {f}")
    arr = pd.read_csv(f, header=None).iloc[:, 0].to_numpy()
    if sim_index < 1 or sim_index > len(arr):
        raise IndexError(f"sim_index {sim_index} out of bounds for {f} (n={len(arr)})")
    return float(arr[sim_index - 1])

def load_residual_series_aligned(csv_path, t_end):
    df = pd.read_csv(csv_path)

    if RESIDUAL_COL_PRIMARY in df.columns:
        col = RESIDUAL_COL_PRIMARY
    elif RESIDUAL_COL_FALLBACK in df.columns:
        col = RESIDUAL_COL_FALLBACK
    else:
        raise ValueError(f"No residual column in {csv_path.name}. Got: {list(df.columns)}")

    if "Time" in df.columns and np.isfinite(t_end):
        df = df[df["Time"] <= t_end]

    vals = df[col].dropna().to_numpy(dtype=np.float32)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)

    return vals.reshape(1, -1, 1)

def normalize_nonzero_inplace(arr):
    vals = arr[0, :, 0]
    nonzero = vals != 0
    if np.any(nonzero):
        mean_abs = float(np.mean(np.abs(vals[nonzero])))
        if mean_abs > 0:
            vals[nonzero] = vals[nonzero] / mean_abs
            arr[0, :, 0] = vals

def unveil_predict_for_one_series(model, residuals_full, ts_len, mult_factor=MULT_FACTOR):
    full_len = residuals_full.shape[1]

    if full_len > ts_len:
        resids = residuals_full[:, -ts_len:, :]
        seq_len = ts_len
    else:
        resids = residuals_full
        seq_len = full_len

    pad_samples = int(ts_len / mult_factor)
    preds = []
    ts_gap = ts_len - seq_len

    for pad_count in range(pad_samples - 1, -1, -1):
        temp_ts = np.zeros((1, ts_len, 1), dtype=np.float32)
        pad_length = mult_factor * pad_count
        zero_range = min(ts_len, pad_length + ts_gap)

        if zero_range == ts_len:
            y_pred = np.zeros((1, 2), dtype=np.float32)
        else:
            if zero_range > 0:
                temp_ts[0, :zero_range, 0] = 0.0
            temp_ts[0, zero_range:, 0] = resids[0, : ts_len - zero_range, 0]
            normalize_nonzero_inplace(temp_ts)

            y_pred = model.predict(temp_ts, verbose=0)
            y_pred = np.asarray(y_pred)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(1, -1)

        preds.append(y_pred.reshape(2,))

    return np.vstack(preds)

def _save_preds_csv(path, arr_2col):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr_2col, delimiter=",", fmt="%.8f")

def _save_ensemble_csv(path, arr_mean_2col):
    df = pd.DataFrame(arr_mean_2col)
    df = df.assign(b=df.iloc[:, [1]].sum(axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=False)

def discover_models(model_dirs, lengths):
    """Search across multiple model directories for matching .keras files."""
    discovered = {lt: [] for lt in lengths}
    for length_tag in lengths:
        for (x, y) in BURY_MODELS:
            filename = f"final_model_{x}_{y}_{length_tag}.keras"
            for mdir in model_dirs:
                cand = mdir / filename
                if cand.exists():
                    discovered[length_tag].append((x, y, mdir))
                    break
    return discovered

def report_models(discovered, lengths):
    print("[models] discovery report:")
    for lt in lengths:
        entries = discovered.get(lt, [])
        pairs = [(a, b) for (a, b, _) in entries]
        head = ", ".join([f"({a},{b})" for (a, b) in pairs[:8]])
        more = " ..." if len(pairs) > 8 else ""
        print(f"  - {lt}: found {len(pairs)} models: {head}{more}")
    for lt in lengths:
        pairs = [(a, b) for (a, b, _) in discovered[lt]]
        have_11 = (1, 1) in pairs
        have_12 = (1, 2) in pairs
        print(f"  - {lt}: Chak presence — (1,1)={have_11} (1,2)={have_12}")


def main():
    csvs = list_residual_csvs(SIM_RESID_DIR)
    if not csvs:
        print(f"No residual CSVs found in: {SIM_RESID_DIR}", file=sys.stderr)
        sys.exit(1)

    sim_tags = [p.stem for p in csvs]
    print(f"Found {len(csvs)} residual files.")

    if SAMPLE_N is not None:
        csvs = csvs[:SAMPLE_N]
        sim_tags = [p.stem for p in csvs]
        print(f"[debug] Limiting to first {SAMPLE_N} sims.")

    discovered = discover_models(MODEL_DIRS, LENGTHS)
    report_models(discovered, LENGTHS)

    # map each sim to its end time
    t_end_list = []
    for stem in sim_tags:
        b, i = parse_sim_tag(stem)
        t_end = load_t_outbreak(LABELS_DIR, b, i)
        t_end_list.append(t_end)

    def get_series(idx):
        return load_residual_series_aligned(csvs[idx], t_end_list[idx])

    for length_tag, ts_len in LENGTH_TAG_TO_LEN.items():
        entries = discovered.get(length_tag, [])
        if not entries:
            print(f"[warn] No models discovered for {length_tag}. Skipping.")
            continue

        pad_samples = int(ts_len / MULT_FACTOR)
        print(f"\n=== Processing {length_tag} (ts_len={ts_len}, pad_samples={pad_samples}, models={len(entries)}) ===")

        N = len(csvs)
        bury_sum = np.zeros((N, pad_samples, 2), dtype=np.float64)
        chak_sum = np.zeros((N, pad_samples, 2), dtype=np.float64)
        bury_count = 0
        chak_count = 0

        for (x, y, mdir) in entries:
            model_filename = f"final_model_{x}_{y}_{length_tag}.keras"
            model_path = mdir / model_filename

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            print(f"[{length_tag}] Loading model: {model_path.name} from {mdir.name}")
            model = load_model(model_path, compile=False)

            for idx, stem in enumerate(sim_tags):
                series = get_series(idx)
                sim_tag = stem
                if (idx + 1) % 500 == 0 or idx + 1 == len(sim_tags):
                    print(f"[{length_tag}] model ({x},{y}): {idx + 1}/{len(sim_tags)}", flush=True)

                bury_pred_dir = OUT_BASE / sim_tag / "bury_compare" / length_tag
                bury_pred_file = bury_pred_dir / f"y_pred_{x}_{y}.csv"

                if SKIP_IF_EXISTS and bury_pred_file.exists():
                    preds = np.loadtxt(bury_pred_file, delimiter=",", dtype=np.float32)
                    if preds.ndim == 1:
                        preds = preds.reshape(-1, 2)
                else:
                    preds = unveil_predict_for_one_series(model, series, ts_len)
                    _save_preds_csv(bury_pred_file, preds)

                if (x, y) in CHAK_MODELS:
                    chak_pred_dir = OUT_BASE / sim_tag / "chakraborty_compare" / length_tag
                    chak_pred_file = chak_pred_dir / f"y_pred_{x}_{y}.csv"
                    if not (SKIP_IF_EXISTS and chak_pred_file.exists()):
                        _save_preds_csv(chak_pred_file, preds)

                bury_sum[idx] += preds
                if (x, y) in CHAK_MODELS:
                    chak_sum[idx] += preds

            keras.backend.clear_session()
            del model
            gc.collect()

            bury_count += 1
            if (x, y) in CHAK_MODELS:
                chak_count += 1

    
        print(f"[{length_tag}] Writing ensemble means (bury_count={bury_count}, chak_count={chak_count})")
        for idx, sim_tag in enumerate(sim_tags):
            if bury_count > 0:
                bury_mean = (bury_sum[idx] / float(bury_count)).astype(np.float32)
                bury_ens_file = OUT_BASE / sim_tag / "bury_compare" / f"ensemble_trend_probs_bury_compare_len{ts_len}.csv"
                _save_ensemble_csv(bury_ens_file, bury_mean)

            if chak_count > 0:
                chak_mean = (chak_sum[idx] / float(chak_count)).astype(np.float32)
                chak_ens_file = OUT_BASE / sim_tag / "chakraborty_compare" / f"ensemble_trend_probs_chakraborty_compare_len{ts_len}.csv"
                _save_ensemble_csv(chak_ens_file, chak_mean)

        print(f"=== Done {length_tag} ===")

    print("\nAll predictions complete.")
    print(f"Output root: {OUT_BASE.resolve()}")

if __name__ == "__main__":
    main()
