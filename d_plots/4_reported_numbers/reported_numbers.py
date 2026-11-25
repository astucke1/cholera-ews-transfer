"""
reported_numbers.py -- the numbers quoted in the Results and Discussion that are not read directly off a figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results"

def scores(name, rule, W):
    return pd.read_csv(RES / "scores" / f"{name}_{rule}_W{W}.csv")

def ews(rule, indicator):
    d = pd.read_csv(RES / "classical_ews" / f"{rule}_rule" / "ews_scores_cumulative_tau.csv")
    d = d[d.indicator == indicator].dropna(subset=["tau", "truth"])
    return d.rename(columns={"truth": "y_true", "tau": "p"})

both = pd.concat([pd.read_csv(RES / "labels" / "both_rules" / f"crossing_days_both_rules_batch{b}.csv").assign(batch=b)
                  for b in range(1, 21)]).set_index(["batch", "sim"])

def attach(d, col):
    idx = pd.MultiIndex.from_arrays([d.batch, d.sim])
    return both.loc[idx, col].to_numpy()

def se_auc(A, n1, n0):
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0))

ramp = [("Cholera-trained ensemble, W=100", scores("cholera_trained", "ramp", 100), True),
        ("Cholera-trained ensemble, W=500", scores("cholera_trained", "ramp", 500), True),
        ("Lag-1 autocorrelation", ews("ramp", "ac"), False),
        ("Variance", ews("ramp", "variance"), False),
        ("SIDATR, W=100", scores("sidatr", "ramp", 100), True),
        ("SIDATR, W=500", scores("sidatr", "ramp", 500), True)]

print("1-3. Held-out SEIARB, slow-parameter (ramp) rule, last five revelation steps pooled")
print(f"  {'indicator':34s} {'AUC':>6} {'acc':>6} {'prec':>6} {'recall':>6} {'SE':>7} {'max batch dev':>14}")
for name, d, prob in ramp:
    y, p = d.y_true.to_numpy().astype(int), d.p.to_numpy()
    A = roc_auc_score(y, p)
    n1 = d[d.y_true == 1].groupby(["batch", "sim"]).ngroups
    n0 = d[d.y_true == 0].groupby(["batch", "sim"]).ngroups
    dev = max(abs(roc_auc_score(g.y_true, g.p) - A) for _, g in d.groupby("batch"))
    if prob:
        c = p >= 0.5
        acc, prec, rec = (c == y).mean(), y[c].mean(), c[y == 1].mean()
        print(f"  {name:34s} {A:6.3f} {acc:6.3f} {prec:6.3f} {rec:6.3f} {se_auc(A, n1, n0):7.4f} {dev:14.3f}")
    else:
        print(f"  {name:34s} {A:6.3f} {'--':>6} {'--':>6} {'--':>6} {se_auc(A, n1, n0):7.4f} {dev:14.3f}")

print("\n4. Label rules")
for rule in ("ramp", "index"):
    lab = both[f"label_{rule}_rule"]
    t = both[f"crossing_day_{rule}_rule"][lab == 1]
    print(f"  {rule:5s} rule: {int(lab.sum()):,} transition / {int((lab == 0).sum()):,} null;"
          f" transition inputs padded at W=100: {int((t < 100).sum()):,}")
omitted = 20000 - ews("index", "ac").groupby(["batch", "sim"]).ngroups
print(f"  simulations omitted by the cumulative tau scorer under the index rule: {omitted}")
rt = both[both.label_ramp_rule == 1]
lead = rt.crossing_day_ramp_rule - rt.crossing_day_index_rule
print(f"  ramp-transition runs whose seasonal-index crossing comes first: {(lead > 0).mean():.1%},"
      f" median lead {lead.median():.0f} days")

print("\n5. Seasonal-index rule, cholera-trained ensemble, W=100")
d = scores("cholera_trained", "index", 100)
d["ramp_label"] = attach(d, "label_ramp_rule")
call = d.p >= 0.5
for desc, m in (("index transition, ramp later crosses", (d.y_true == 1) & (d.ramp_label == 1)),
                ("index transition, ramp never crosses", (d.y_true == 1) & (d.ramp_label == 0)),
                ("index null", d.y_true == 0)):
    print(f"  {desc:40s} called transition: {call[m].mean():.1%}")
m = d.y_true == 1
print(f"  AUC separating the two transition groups: {roc_auc_score(d.ramp_label[m], d.p[m]):.3f}")
y, c = d.y_true.to_numpy(), call.to_numpy()
print(f"  precision {y[c].mean():.3f}, recall {c[y == 1].mean():.3f}")

print("\n6. Decomposition at W=100: AUC with inputs and labels changed one at a time")
for name in ("sidatr", "cholera_trained"):
    out = {}
    for inp in ("ramp", "index"):
        d = scores(name, inp, 100)
        for labrule in ("ramp", "index"):
            out[(inp, labrule)] = roc_auc_score(attach(d, f"label_{labrule}_rule"), d.p)
    print(f"  {name:16s} slow-parameter inputs: ramp labels {out[('ramp','ramp')]:.3f}, index labels {out[('ramp','index')]:.3f}"
          f" | seasonal-index inputs: ramp labels {out[('index','ramp')]:.3f}, index labels {out[('index','index')]:.3f}")
d = scores("sidatr", "index", 100); c = d.p >= 0.5
print(f"  SIDATR calls transition: ramp rule {(scores('sidatr','ramp',100).p >= 0.5).mean():.1%},"
      f" index rule {c.mean():.1%}; index-rule accuracy {(c == d.y_true).mean():.3f}"
      f" vs all-transition {d.y_true.mean():.3f}")
