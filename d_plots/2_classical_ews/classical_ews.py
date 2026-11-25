####################################################################
# classical_ews.py -- classical early warning indicators on the SEIARB set (pipeline step 5)
####################################################################

import sys
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as metrics
from pathlib import Path
import warnings
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent

_parser = argparse.ArgumentParser(description="Classical early warning indicators on the SEIARB residual series.")
_parser.add_argument("--label-rule", choices=("ramp", "index"), default="ramp",
                     help="SEIARB transition label rule (default: ramp, the rule behind the main results)")
SEIIRB_LABEL_RULE = _parser.parse_args().label_rule
SEIIRB_OUTPUT = REPO_ROOT / "a_simulations" / "3_seiirb" / "output" / f"SEIIRB_{SEIIRB_LABEL_RULE}_rule"
RESIDS_DIR    = SEIIRB_OUTPUT / "output_resids"
LABELS_DIR    = SEIIRB_OUTPUT / "output_labels"

OUTPUT_DIR = SCRIPT_DIR / "output" / f"classical_ews_results_{SEIIRB_LABEL_RULE}_rule"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BATCHES = 20
SIMS_PER_BATCH = 1000

ROLLING_WINDOW_FRAC = 0.25
ML_SPACING = 10
N_PREDICTIONS = 5
PRED_INTERVAL = (0.8, 1.0)


def compute_rolling_ews(residuals, window_size):
    var_series = residuals.rolling(window=window_size, min_periods=window_size).var()
    shifted = residuals.shift(1)
    ac_series = residuals.rolling(window=window_size, min_periods=window_size).corr(shifted)
    return var_series, ac_series


def ktau_compute_cumulative(series, t_fin):
    """
    cumulative kendall tau from second non-null value to t_fin.
    matches chakraborty's index[1] start behavior.
    """
    valid = series[pd.notnull(series)]
    if len(valid) < 3:
        return np.nan
    t_start = valid.index[1]
    if t_fin < t_start:
        return np.nan
    series_reduced = series.loc[t_start:t_fin].dropna()
    if len(series_reduced) < 2:
        return np.nan
    ktau, _ = stats.kendalltau(series_reduced.index.values, series_reduced.values)
    return ktau


def ktau_cumulative_series(series, t_res=10):
    """cumulative tau evaluated every t_res steps."""
    t_vals = series.index[::t_res]
    ktau_vals = []
    ktau_times = []
    for t in t_vals:
        ktau = ktau_compute_cumulative(series, t)
        if not np.isnan(ktau):
            ktau_vals.append(ktau)
            ktau_times.append(t)
    if len(ktau_vals) == 0:
        return pd.Series(dtype=float)
    return pd.Series(ktau_vals, index=ktau_times)


def main():
    t0 = time.time()
    records = []
    skipped_missing = 0
    skipped_short = 0

    for batch in range(1, N_BATCHES + 1):
        batch_t0 = time.time()

        if not (LABELS_DIR / f"label_batch{batch}.csv").exists():   # stop at the last batch present
            print(f"label_batch{batch}.csv not found; stopping at batch {batch-1}")
            break
        label_df = pd.read_csv(LABELS_DIR / f"label_batch{batch}.csv", header=None)
        labels = label_df.iloc[:, 0].values

        outbreak_df = pd.read_csv(LABELS_DIR / f"outbreak_times_batch{batch}.csv", header=None)
        outbreak_times = outbreak_df.iloc[:, 0].values

        for sim_idx in range(len(labels)):
            sim_num = sim_idx + 1
            label = int(labels[sim_idx])
            t_outbreak = float(outbreak_times[sim_idx])

            resid_file = RESIDS_DIR / f"resids_batch{batch}_sim{sim_num}.csv"
            if not resid_file.exists():
                skipped_missing += 1
                continue

            df_resid = pd.read_csv(resid_file)
            df_resid = df_resid[df_resid['Time'] <= t_outbreak]

            if len(df_resid) < 10:
                skipped_short += 1
                continue

            residuals = pd.Series(
                df_resid['residuals'].values,
                index=df_resid['Time'].values
            )

            # rolling ews
            n_points = len(residuals)
            window_size = max(2, int(ROLLING_WINDOW_FRAC * n_points))

            var_series, ac_series = compute_rolling_ews(residuals, window_size)

            t_end = residuals.index[-1]
            t_start = residuals.index[0]
            t_pred_start = t_start + (t_end - t_start) * PRED_INTERVAL[0]

            # cumulative tau, filtered to last 20%, last 5 values
            for indicator_name, ews_series in [('variance', var_series), ('ac', ac_series)]:
                ktau_series_result = ktau_cumulative_series(ews_series, t_res=ML_SPACING)

                if len(ktau_series_result) > 0:
                    in_window = ktau_series_result[
                        (ktau_series_result.index >= t_pred_start) &
                        (ktau_series_result.index <= t_end)
                    ].tail(N_PREDICTIONS)

                    for tau_val in in_window.values:
                        if np.isfinite(tau_val):
                            records.append({
                                'tau': tau_val,
                                'truth': label,
                                'indicator': indicator_name,
                                'batch': batch,
                                'sim': sim_num,
                            })

        batch_elapsed = time.time() - batch_t0
        print(f"batch {batch:2d}/{N_BATCHES} done  ({batch_elapsed:.1f}s)")

    # compute auc
    print("\n" + "=" * 60)
    print("CLASSICAL EWS AUC — CHAKRABORTY CUMULATIVE TAU")
    print("=" * 60)

    df = pd.DataFrame(records)

    if len(df) > 0:
        for ind in ['variance', 'ac']:
            sub = df[df['indicator'] == ind]
            if len(sub) == 0 or sub['truth'].nunique() < 2:
                print(f"  {ind.upper()}: insufficient data")
                continue

            auc = metrics.roc_auc_score(sub['truth'], sub['tau'])
            n_sims = sub.groupby(['batch', 'sim']).ngroups
            n_pos = int(sub['truth'].sum())
            n_neg = len(sub) - n_pos
            print(f"  {ind.upper()}:  AUC = {auc:.4f}  "
                  f"(n={n_sims} sims, {n_pos} pos, {n_neg} neg, "
                  f"{len(sub)} total scores)")

        df.to_csv(OUTPUT_DIR / "ews_scores_cumulative_tau.csv", index=False)
        print(f"\nScores saved to {OUTPUT_DIR}")

    if skipped_missing > 0:
        print(f"\n[note] {skipped_missing} missing residual files skipped")
    if skipped_short > 0:
        print(f"[note] {skipped_short} series too short, skipped")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
