####################################################################
# evaluate_predictions.py -- final-revelation-step metrics for the cholera-trained ensembles (pipeline step 7)
####################################################################

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

# ---- path convention ----
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent


_parser = argparse.ArgumentParser(description="Evaluate the cholera-trained predictions against the SEIARB labels.")
_parser.add_argument("--label-rule", choices=("ramp", "index"), default="ramp",
                     help="SEIARB transition label rule (default: ramp, the rule behind the main results)")
SEIIRB_LABEL_RULE = _parser.parse_args().label_rule
SEIIRB_OUTPUT = REPO_ROOT / "a_simulations" / "3_seiirb" / "output" / f"SEIIRB_{SEIIRB_LABEL_RULE}_rule"
LABELS_DIR    = SEIIRB_OUTPUT / "output_labels"

PRED_BASE = REPO_ROOT / "c_dl_apply" / "cholera_trained" / "output" / f"predictions_{SEIIRB_LABEL_RULE}_rule"
r
OUT_DIR = SCRIPT_DIR / "output" / f"evaluation_{SEIIRB_LABEL_RULE}_rule"

FAMILIES = ["bury_compare", "chakraborty_compare"]
LENGTHS = [100, 500]

MULT_FACTOR = 10

SIM_TAG_RE = re.compile(r"resids_batch(\d+)_sim(\d+)$")


def parse_sim_tag(sim_tag):
    m = SIM_TAG_RE.match(sim_tag)
    if not m:
        raise ValueError(f"Unrecognized sim_tag: {sim_tag}")
    return int(m.group(1)), int(m.group(2))


def load_all_labels(labels_dir, max_batch=20):
    labels = {}
    for batch in range(1, max_batch + 1):
        label_file = labels_dir / f"label_batch{batch}.csv"
        if not label_file.exists():
            print(f"[warn] Missing: {label_file}")
            continue
        df = pd.read_csv(label_file, header=None)
        arr = df.iloc[:, 0].to_numpy()

        start_idx = 0
        if not np.issubdtype(type(arr[0]), np.number):
            try:
                int(arr[0])
            except (ValueError, TypeError):
                start_idx = 1

        for i, val in enumerate(arr[start_idx:], start=1):
            labels[(batch, i)] = int(val)
    return labels


def list_prediction_dirs(pred_base):
    return sorted(
        [d.name for d in pred_base.iterdir() if d.is_dir() and d.name.startswith("resids_batch")],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
    )


def load_ensemble_preds(pred_base, sim_tag, family, ts_len):
    fname = f"ensemble_trend_probs_{family}_len{ts_len}.csv"
    fpath = pred_base / sim_tag / family / fname
    if not fpath.exists():
        return None
    arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def compute_metrics_at_step(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        m["roc_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["tn"] = int(tn)
    m["fp"] = int(fp)
    m["fn"] = int(fn)
    m["tp"] = int(tp)
    return m


def evaluate_family_length(pred_base, labels, sim_tags, family, ts_len):
    pad_samples = ts_len // MULT_FACTOR

    all_preds = []
    all_labels = []
    valid_sim_tags = []

    for sim_tag in sim_tags:
        batch, sim_idx = parse_sim_tag(sim_tag)
        if (batch, sim_idx) not in labels:
            continue

        preds = load_ensemble_preds(pred_base, sim_tag, family, ts_len)
        if preds is None:
            continue

        trans_prob = preds[:, 1]
        if len(trans_prob) != pad_samples:
            print(f"[warn] {sim_tag}/{family}/len{ts_len}: expected {pad_samples} steps, got {len(trans_prob)}")
            continue

        all_preds.append(trans_prob)
        all_labels.append(labels[(batch, sim_idx)])
        valid_sim_tags.append(sim_tag)

    if not all_preds:
        print(f"[warn] No valid predictions for {family}/len{ts_len}")
        return pd.DataFrame(), pd.DataFrame()

    preds_matrix = np.vstack(all_preds)
    labels_arr = np.array(all_labels)

    n_sims = len(labels_arr)
    n_null = np.sum(labels_arr == 0)
    n_trans = np.sum(labels_arr == 1)
    print(f"[{family}/len{ts_len}] Evaluating {n_sims} sims ({n_null} null, {n_trans} transition)")

    # metrics at each unveiling step
    step_metrics = []
    for step in range(pad_samples):
        y_prob = preds_matrix[:, step]
        m = compute_metrics_at_step(labels_arr, y_prob)
        m["step"] = step
        m["revealed_frac"] = (pad_samples - step) / pad_samples
        step_metrics.append(m)

    metrics_df = pd.DataFrame(step_metrics)
    metrics_df["family"] = family
    metrics_df["ts_len"] = ts_len

    # per-simulation results at final step
    final_probs = preds_matrix[:, -1]
    final_preds = (final_probs >= 0.5).astype(int)

    sim_results_df = pd.DataFrame({
        "sim_tag": valid_sim_tags,
        "y_true": labels_arr,
        "y_prob": final_probs,
        "y_pred": final_preds,
        "correct": (final_preds == labels_arr).astype(int),
    })
    sim_results_df["family"] = family
    sim_results_df["ts_len"] = ts_len

    return metrics_df, sim_results_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth labels...")
    labels = load_all_labels(LABELS_DIR)
    print(f"Loaded {len(labels)} labels")
    n_null = sum(1 for v in labels.values() if v == 0)
    n_trans = sum(1 for v in labels.values() if v == 1)
    print(f"  Class distribution: {n_null} null, {n_trans} transition")

    sim_tags = list_prediction_dirs(PRED_BASE)
    print(f"Found {len(sim_tags)} prediction directories")

    if not sim_tags:
        print("No predictions found. Exiting.")
        sys.exit(1)

    all_metrics = []
    all_sim_results = []

    for family in FAMILIES:
        for ts_len in LENGTHS:
            metrics_df, sim_results_df = evaluate_family_length(
                PRED_BASE, labels, sim_tags, family, ts_len
            )
            if not metrics_df.empty:
                all_metrics.append(metrics_df)
                all_sim_results.append(sim_results_df)

    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_metrics.to_csv(OUT_DIR / "metrics_by_step.csv", index=False)

        combined_sim_results = pd.concat(all_sim_results, ignore_index=True)
        combined_sim_results.to_csv(OUT_DIR / "sim_results_final_step.csv", index=False)

        final_metrics = combined_metrics[
            combined_metrics["step"] == combined_metrics.groupby(["family", "ts_len"])["step"].transform("max")
        ]

        summary = final_metrics[[
            "family", "ts_len", "accuracy", "f1", "precision", "recall", "roc_auc",
            "tp", "fp", "tn", "fn"
        ]]
        summary.to_csv(OUT_DIR / "summary_final_step.csv", index=False)

        print("\n" + "=" * 70)
        print("FINAL STEP METRICS (fully revealed)")
        print("=" * 70)
        print(summary.to_string(index=False))
        print("=" * 70)
        print(f"\nResults saved to: {OUT_DIR.resolve()}")
    else:
        print("No metrics computed. Check that predictions exist.")


if __name__ == "__main__":
    main()
