"""
make_figures.py -- Figures 2 and 3 of the paper and the AUC / confusion-matrix tables behind them.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent.parent
RESULTS    = ROOT / "results"
SCORES     = RESULTS / "scores"
OUT        = RESULTS / "figures"
SCORES.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

EVAL_PIPE = ROOT / "d_plots" / "1_evaluate_predictions" / "output"
EWS_PIPE  = ROOT / "d_plots" / "2_classical_ews" / "output"
PRED_PIPE = ROOT / "c_dl_apply" / "cholera_trained" / "output"

COL = {
    "dl100":  "tab:cyan",   # cholera-trained, W = 100
    "dl500":  "tab:blue",   # cholera-trained, W = 500
    "sid100": "tab:orange",   # SIDATR, W = 100
    "sid500": "tab:red",   # SIDATR, W = 500
    "ac":     "tab:green",   # lag-1 autocorrelation
    "var":    "tab:purple",   # variance
}
LAB = {
    "dl100":  "Cholera-trained, W = 100",
    "dl500":  "Cholera-trained, W = 500",
    "sid100": "SIDATR, W = 100",
    "sid500": "SIDATR, W = 500",
    "ac":     "Lag-1 autocorrelation",
    "var":    "Variance",
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150})

def _last5(path):
    """last five transition probabilities from a prediction CSV (columns: P(null), P(transition))."""
    with open(path) as fh:
        rows = fh.read().strip().split("\n")[-5:]
    return np.array([float(r.split(",")[1]) for r in rows])

def _cached(name, fn):
    """read results/scores/<name>.csv if present; otherwise compute from the pipeline and write it."""
    f = SCORES / f"{name}.csv"
    if f.exists():
        d = pd.read_csv(f)
        return d.y_true.to_numpy().astype(int), d.p.to_numpy().astype(float)
    y, p, ids = fn()
    if y is not None:
        pd.DataFrame({"batch": np.repeat(ids[:, 0], 5), "sim": np.repeat(ids[:, 1], 5),
                      "step": np.tile([1, 2, 3, 4, 5], len(ids)), "y_true": y, "p": p}
                     ).to_csv(f, index=False, float_format="%.8f")
    return y, p

def _dl_scores(rule, W, members=("1_1", "1_2")):
    """cholera-trained ensemble from the pipeline outputs: y_true repeated 5x, p at the last five
    revelation steps pooled."""
    listing = EVAL_PIPE / f"evaluation_{rule}_rule" / "sim_results_final_step.csv"
    if not listing.exists():
        listing = RESULTS / "evaluation" / f"{rule}_rule" / "sim_results_final_step.csv"
    pred = PRED_PIPE / f"predictions_{rule}_rule"
    if not pred.exists():
        raise FileNotFoundError(f"no shipped scores for cholera-trained {rule} W={W} and no pipeline output at {pred}; "
                                "run apply_cholera_trained.py first")
    df = pd.read_csv(listing)
    df = df[(df.family == "bury_compare") & (df.ts_len == W)]
    ys, ps, ids = [], [], []
    for tag, yt in zip(df.sim_tag, df.y_true):
        ids.append([int(x) for x in tag.replace("resids_batch", "").split("_sim")])
        d = pred / tag / "bury_compare" / f"length{W}"
        acc = None
        for m in members:
            a = _last5(d / f"y_pred_{m}.csv")
            acc = a if acc is None else acc + a
        ps.append(acc / len(members)); ys.append(int(yt))
    return np.repeat(np.array(ys), 5), np.concatenate(ps), np.array(ids)

def dl_scores(rule, W):
    return _cached(f"cholera_trained_{rule}_W{W}", lambda: _dl_scores(rule, W))

def sidatr_scores(rule, W):
    """SIDATR scores are shipped only (results/scores/sidatr_<rule>_W<W>.csv); None if absent."""
    return _cached(f"sidatr_{rule}_W{W}", lambda: (None, None, None))

def ews_scores(rule, indicator):
    f = EWS_PIPE / f"classical_ews_results_{rule}_rule" / "ews_scores_cumulative_tau.csv"
    if not f.exists():
        f = RESULTS / "classical_ews" / f"{rule}_rule" / "ews_scores_cumulative_tau.csv"
    df = pd.read_csv(f)
    sub = df[df.indicator == indicator].dropna(subset=["tau", "truth"])
    return sub.truth.to_numpy().astype(int), sub.tau.to_numpy().astype(float)

def auc(y, p):
    return roc_auc_score(y, p) if y is not None and len(y) else np.nan

# ---------- Figure 2: ROC, ramp rule ----------
series = [
    ("dl500",  dl_scores("ramp", 500)),
    ("dl100",  dl_scores("ramp", 100)),
    ("ac",     ews_scores("ramp", "ac")),
    ("sid500", sidatr_scores("ramp", 500)),
    ("var",    ews_scores("ramp", "variance")),
    ("sid100", sidatr_scores("ramp", 100)),
]
fig, ax = plt.subplots(figsize=(4.2, 4.2))
fig2_rows = []
for key, (y, p) in series:
    if y is None:
        print(f"[skip] no scores for {LAB[key]}")
        continue
    a = auc(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    ls = "-"
    ax.plot(fpr, tpr, ls, color=COL[key], lw=1.4, label=f"{LAB[key]} ({a:.2f})")
    fig2_rows.append({"indicator": LAB[key], "n": len(y), "AUC": round(a, 4)})
ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":", alpha=0.6)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=7.5, title="AUC, last five revelation steps", title_fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_roc_ramp.png"); fig.savefig(OUT / "fig2_roc_ramp.pdf"); plt.close(fig)
pd.DataFrame(fig2_rows).to_csv(OUT / "fig2_auc_values.csv", index=False)

# ---------- Figure 3: ramp vs index ----------
keys3 = ["dl100", "ac", "var", "sid100"]
getters = {
    "dl100":  lambda r: dl_scores(r, 100),
    "ac":     lambda r: ews_scores(r, "ac"),
    "var":    lambda r: ews_scores(r, "variance"),
    "sid100": lambda r: sidatr_scores(r, 100),
}
vals = {k: {r: auc(*getters[k](r)) for r in ("ramp", "index")} for k in keys3}

fig, (axL, axR1, axR2) = plt.subplots(1, 3, figsize=(9.4, 3.2),
                                      gridspec_kw={"width_ratios": [1.7, 1, 1], "wspace": 0.55})
x = np.arange(len(keys3)); w = 0.36
axL.bar(x - w/2, [vals[k]["ramp"] for k in keys3], w, color=[COL[k] for k in keys3])
axL.bar(x + w/2, [vals[k]["index"] for k in keys3], w, color=[COL[k] for k in keys3], alpha=0.45,
        hatch="///", edgecolor="white")
axL.legend(handles=[Patch(color="#555555", label="Ramp rule"),
                    Patch(facecolor="#555555", alpha=0.45, hatch="///", edgecolor="white", label="Index rule")],
           loc="upper right", fontsize=7.5)
for i, k in enumerate(keys3):
    axL.text(i - w/2, vals[k]["ramp"] + 0.01, f"{vals[k]['ramp']:.2f}", ha="center", va="bottom", fontsize=7)
    axL.text(i + w/2, vals[k]["index"] + 0.01, f"{vals[k]['index']:.2f}", ha="center", va="bottom", fontsize=7)
axL.axhline(0.5, color="k", lw=0.6, ls=":", alpha=0.6)
axL.set_xticks(x); axL.set_xticklabels(["Cholera-\ntrained", "Lag-1 AC", "Variance", "SIDATR"], fontsize=8)
axL.set_ylim(0.3, 1.0); axL.set_ylabel("AUC (W = 100)")
axL.set_title("(a) AUC by label rule", fontsize=9, loc="left")

def cm_panel(ax, y, p, title):
    yhat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])      # rows: true null, true transition
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.9 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["null", "transition"], fontsize=7.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["null", "transition"], fontsize=7.5)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")
    tn, fp, fn, tp = cm.ravel()
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                precision=tp / (tp + fp), recall=tp / (tp + fn))

cm_rows = []
y_i, p_i = sidatr_scores("index", 100)
if y_i is not None:
    cm_rows.append(dict(model="SIDATR W=100", **cm_panel(axR1, y_i, p_i, "(b) SIDATR, index")))
y_i, p_i = dl_scores("index", 100)
cm_rows.append(dict(model="Cholera-trained W=100", **cm_panel(axR2, y_i, p_i, "(c) Cholera-trained, index")))
fig.tight_layout()
fig.savefig(OUT / "fig3_label_rule.png"); fig.savefig(OUT / "fig3_label_rule.pdf"); plt.close(fig)
pd.DataFrame([{"indicator": LAB[k], "ramp_AUC": vals[k]["ramp"], "index_AUC": vals[k]["index"]} for k in keys3]
             ).to_csv(OUT / "fig3_auc_values.csv", index=False)
pd.DataFrame(cm_rows).to_csv(OUT / "fig3_confusion_index_rule.csv", index=False)

print("Figure 2 AUCs (ramp rule, last five revelation steps pooled):")
print(pd.DataFrame(fig2_rows).to_string(index=False))
print("\nFigure 3 AUCs (W = 100):")
for k in keys3:
    print(f"  {LAB[k]:28s} ramp {vals[k]['ramp']:.3f}  index {vals[k]['index']:.3f}")
for r in cm_rows:
    print("  index-rule confusion matrix:", r)
print("\nWrote:", sorted(p.name for p in OUT.iterdir()))
