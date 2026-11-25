import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent
EDATA = REPO / "e_data"
OUT = REPO / "results" / "figures"


# ---------- SEIARB fit to Haiti 2022 weekly incidence ----------
def fig_haiti_fit():
    from scipy.integrate import odeint

    raw = pd.read_csv(EDATA / "choleraHaiti202220232024.csv", skiprows=1)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(columns={"newCase.week": "cases"})
    obs = raw.iloc[39:].reset_index(drop=True)          # weeks 1-39 are zero; week 40 starts 2 Oct 2022
    obs["global_week"] = np.arange(1, len(obs) + 1)
    obs["t"] = (obs["global_week"] - 1) * 7
    K = 52
    obs_fit = obs[obs["global_week"] <= K]

    with open(EDATA / "seiirb_fitted_params.json") as f:
        pdata = json.load(f)
    parms = {**pdata["fixed"], **pdata["fitted"]}

    def cholera_ode(y, t, q):
        S, E, Is, Ia, R, B = y
        beta_b_t = q["beta_b"] * (1 + q["a1"] * np.sin(2 * np.pi * t / 365 + q["phi1"])
                                  + q["a2"] * np.sin(2 * np.pi * t / 182.5 + q["phi2"]))
        lam = q["beta_s"] * Is / q["N"] + q["beta_a"] * Ia / q["N"] + beta_b_t * B / (q["k"] + B)
        return [q["mu"] * q["N"] + q["omega"] * R - lam * S - q["mu"] * S,
                lam * S - (q["sigma"] + q["mu"]) * E,
                q["p"] * q["sigma"] * E - (q["gamma_s"] + q["alpha_s"] + q["mu"]) * Is,
                (1 - q["p"]) * q["sigma"] * E - (q["gamma_a"] + q["mu"]) * Ia,
                q["gamma_s"] * Is + q["gamma_a"] * Ia - (q["omega"] + q["mu"]) * R,
                q["eta_s"] * Is + q["eta_a"] * Ia - q["delta"] * B]

    N = parms["N"]
    state0 = [N - 800 - 7200 - 306, 306, 800, 7200, 0, 1000]
    times = np.arange(0, obs["t"].max() + 1)
    sol = odeint(cholera_ode, state0, times, args=(parms,))
    daily_new_symptomatic = parms["p"] * parms["sigma"] * sol[:, 1]
    week_idx = (times // 7).astype(int) + 1
    df_w = pd.DataFrame({"week": week_idx, "predicted": daily_new_symptomatic}).groupby("week").sum().reset_index()
    df_pub = obs_fit.merge(df_w, left_on="global_week", right_on="week", how="left")

    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 14, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 11}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(df_pub["global_week"], df_pub["cases"], color="tab:blue", alpha=0.55, width=0.8,
               edgecolor="tab:blue", linewidth=0.3, zorder=2, hatch="//")
        ax.plot(df_pub["global_week"], df_pub["predicted"], color="tab:red", lw=2.5, alpha=0.9, zorder=3)
        ax.set_xlabel("Week since Oct 2, 2022")
        ax.set_ylabel("Weekly new cases")
        ax.legend(loc="upper right", handles=[
            Line2D([0], [0], marker="s", ls="", color="tab:blue", alpha=0.55, markersize=7, label="Observed"),
            Line2D([0], [0], color="tab:red", lw=2.5, label="Fitted SEIARB")])
        fig.tight_layout()
        fig.savefig(OUT / "fig1_seiirb_haiti_fit.pdf", bbox_inches="tight")
        fig.savefig(OUT / "fig1_seiirb_haiti_fit.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    print("  [haiti_fit] fig1_seiirb_haiti_fit.pdf/.png")


# ---------- labeling schematic ----------
def fig_schematic():
    # SEIARB parameters (Haiti 2022), identical to a_simulations/3_seiirb/simulate_seiirb.py
    N = 11_637_398; mu = 4.30e-5; sigma = 0.5; p = 0.10; gamma_s = 0.20; gamma_a = 0.333; alpha_s = 0.0062
    kappa = 1e6; eta_s = 10.0; eta_a = 1.0; delta = 0.3
    beta_s = 2.84e-11; beta_a = 9.29e-8; beta_0 = 3.748e-3
    A1, PHI1, A2, PHI2 = 0.131, 2.27, 0.235, 0.983
    tmax, W = 1500.0, 500
    t = np.arange(0.0, tmax + 1, 1.0)
    C_s = (p * sigma) / ((sigma + mu) * (gamma_s + alpha_s + mu))
    C_a = ((1.0 - p) * sigma) / ((sigma + mu) * (gamma_a + mu))
    beta_b_c = (1.0 - (beta_s * C_s + beta_a * C_a)) / ((N / (kappa * delta)) * (eta_s * C_s + eta_a * C_a))
    m_c = beta_b_c / beta_0
    q = 1.0 + A1 * np.sin(2 * np.pi * t / 365.0 + PHI1) + A2 * np.sin(2 * np.pi * t / 182.5 + PHI2)

    # two illustrative ramps within the sampling ranges of simulate_seiirb.py
    runs = {"Transition": dict(m0=0.55 * m_c, m1=1.5 * (0.45 * m_c) / tmax, color="tab:purple", ls="-"),
            "Null":       dict(m0=0.50 * m_c, m1=0.4 * (0.50 * m_c) / tmax, color="tab:green", ls="--")}
    for r in runs.values():
        r["ramp"] = (r["m0"] + r["m1"] * t) / m_c
        r["Ridx"] = r["ramp"] * q
        c = np.where(r["ramp"] >= 1.0)[0]; r["t_ramp"] = float(t[c[0]]) if c.size else None
        c = np.where(r["Ridx"] >= 1.0)[0]; r["t_index"] = float(t[c[0]]) if c.size else None
    tr, nu = runs["Transition"], runs["Null"]
    THRESH = dict(color="tab:brown", lw=2.0)
    TC = dict(color="tab:red", lw=1.4, ls=":")

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 11,
                         "legend.frameon": True, "legend.framealpha": 0.9}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)

        def panel(ax, key, label, lines, window_end, title):
            for r in runs.values():
                ax.plot(t, r[key], color=r["color"], ls=r["ls"], lw=1.8)
            if key == "Ridx":
                ax.plot(t, tr["ramp"], color=tr["color"], lw=1.0, alpha=0.55)   # underlying ramp
            ax.axhline(1.0, **THRESH)
            for day in lines:
                if day is not None:
                    ax.axvline(day, **TC)
            if window_end is not None:
                ax.axvspan(window_end - W, window_end, color="khaki", alpha=0.35, lw=0)
            ax.text(0.03, 0.93, label, transform=ax.transAxes, fontsize=13, style="italic")
            ax.set_title(title, fontsize=10.5, pad=8)
            ax.set_xlabel("Time (days)"); ax.set_xlim(0, tmax); ax.set_ylim(0.0, 1.3)

        panel(axes[0], "ramp", "(a)", [tr["t_ramp"]], tr["t_ramp"], "Chakraborty et al. (2024): linear ramp, no forcing")
        panel(axes[1], "Ridx", "(b)", [tr["t_ramp"]], tr["t_ramp"], "This study, slow-parameter rule")
        panel(axes[2], "Ridx", "(c)", [tr["t_index"], nu["t_index"]], tr["t_index"], "Same runs, seasonal-index rule")
        ti = tr["t_index"]
        axes[2].plot(ti, float(np.interp(ti, t, tr["ramp"])), "o", color=tr["color"], ms=6, mfc="white", mew=1.6, zorder=5)
        axes[0].set_ylabel("Control parameter / threshold")
        axes[1].set_ylabel(r"$\mathcal{R}_{\mathrm{idx}}(t)$")
        axes[2].set_ylabel(r"$\mathcal{R}_{\mathrm{idx}}(t)$")
        for ax in axes[1:]:
            ax.yaxis.set_tick_params(labelleft=True)
        axes[0].legend(handles=[
            Line2D([0], [0], color=tr["color"], lw=1.8, label="Transition"),
            Line2D([0], [0], color=nu["color"], lw=1.8, ls="--", label="Null"),
            Line2D([0], [0], label="Threshold", **THRESH),
            Line2D([0], [0], label=r"$t_c$ (labelled transition)", **TC),
            Line2D([0], [0], color=tr["color"], lw=1.0, alpha=0.55, label="Underlying ramp (b, c)"),
            Line2D([0], [0], marker="o", color="none", mec=tr["color"], mfc="white", mew=1.6, ms=6,
                   label="Ramp when index first reaches 1 (c)"),
            Patch(facecolor="khaki", alpha=0.5, label=r"$W=500$ window")], loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig1_schematic.pdf"); fig.savefig(OUT / "fig1_schematic.png", dpi=200)
        plt.close(fig)
    print(f"  [schematic] fig1_schematic.pdf/.png (slow-parameter crossing day {tr['t_ramp']:.0f},"
          f" seasonal-index crossing day {ti:.0f}, ramp {np.interp(ti, t, tr['ramp']):.2f} of threshold)")


# ---------- ROC and label-rule figures ----------
def fig_roc_label_rule():
    with plt.rc_context():
        runpy.run_path(str(REPO / "d_plots" / "3_figures" / "make_figures.py"), run_name="__main__")
    print("  [roc_label_rule] fig2_roc_ramp.pdf/.png, fig3_label_rule.pdf/.png")


ALL = {"haiti_fit": fig_haiti_fit, "schematic": fig_schematic, "roc_label_rule": fig_roc_label_rule}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(ALL), help="comma-separated subset of: " + ", ".join(ALL))
    args = ap.parse_args()
    chosen = [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in chosen if s not in ALL]
    if bad:
        raise SystemExit(f"unknown figure(s) {bad}; choose from {list(ALL)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        ALL[name]()
    print(f"Done. Figures in {OUT}")
