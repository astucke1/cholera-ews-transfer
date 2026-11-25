####################################################################
# simulate_seiirb.py -- SEIARB held-out test-set generation (pipeline step 3)
#
# 20 batches x 1,000 stochastic SEIARB simulations (called SEIIRB in the code: two infected
# classes, symptomatic Is and asymptomatic Ia) with dual-seasonal forcing and a linearly
# ramping environmental-transmission scalar, parameterised to the 2022 Haiti cholera outbreak.
# These simulations are never used for training. Each run is labelled under both transition
# rules described below and written to output/SEIIRB_ramp_rule/ and output/SEIIRB_index_rule/.
# Runtime: roughly 23 hours.
#
# The SEIARB model:
#   dS/dt  = mu*N + omega*R - lambda(t)*S - mu*S
#   dE/dt  = lambda(t)*S - (sigma + mu)*E
#   dIs/dt = p*sigma*E - (gamma_s + alpha_s + mu)*Is
#   dIa/dt = (1-p)*sigma*E - (gamma_a + mu)*Ia
#   dR/dt  = gamma_s*Is + gamma_a*Ia - (omega + mu)*R
#   dB/dt  = eta_s*Is + eta_a*Ia - delta*B
#
# where lambda(t) = beta_s*Is/N + beta_a*Ia/N + beta_b(t)*B/(kappa+B)
# and beta_b(t) = m(t) * beta_0 * q(t)
#   m(t) = m0 + m1*t  is the ramping scalar (control parameter)
#   q(t) is dual-seasonal forcing with per-realization jitter
#
####################################################################

import os
import gc
import numpy as np
import pandas as pd
import ewstools
from numpy.random import default_rng


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR   = os.path.join(REPO_ROOT, "e_data")

LABEL_RULES = {
    "index": {
        "root": os.path.join(SCRIPT_DIR, "output", "SEIIRB_index_rule"),
        "description": ("INDEX RULE (seasonal-index rule, sensitivity analysis). transition = first day the seasonal index "
                        "beta_b(t)/beta_b_c = (m0 + m1*t)*q(t)/(m_c) reaches 1. A seasonal peak can trigger it, "
                        "so a ramp-null run can be labeled transition. Class balance is NOT 50/50."),
    },
    "ramp": {
        "root": os.path.join(SCRIPT_DIR, "output", "SEIIRB_ramp_rule"),
        "description": ("RAMP RULE (slow-parameter rule, C4; Chakraborty et al. 2024 analogue). transition = the day the ramp m0 + m1*t "
                        "reaches m_c, the cycle-averaged threshold; seasonal forcing does not enter the label. "
                        "Class balance is 50/50 by construction. Crossing day = (m_c - m0)/m1."),
    },
}
for rule, cfg in LABEL_RULES.items():
    cfg["sims"]   = os.path.join(cfg["root"], "output_sims")
    cfg["resids"] = os.path.join(cfg["root"], "output_resids")
    cfg["labels"] = os.path.join(cfg["root"], "output_labels")
    for folder in (cfg["sims"], cfg["resids"], cfg["labels"]):
        os.makedirs(folder, exist_ok=True)
    with open(os.path.join(cfg["root"], "LABEL_RULE.txt"), "w") as fh:
        fh.write(cfg["description"] + "\n\nFiles: output_sims/tseries_batch{b}_sim{i}.csv, "
                 "output_resids/resids_batch{b}_sim{i}.csv, output_labels/label_batch{b}.csv (0/1), "
                 "output_labels/outbreak_times_batch{b}.csv (crossing day; tmax if none).\n")
both_rules_folder = os.path.join(SCRIPT_DIR, "output", "SEIIRB_both_rules_comparison")
os.makedirs(both_rules_folder, exist_ok=True)


n_batches     = 20
sim_per_batch = 1000


dt    = 0.01
t0    = 0.0
tmax  = 1500.0
tburn = 100.0
dt2   = 1.0
rw    = 0.25
span  = 0.2

t = np.arange(t0, tmax, dt)
step_daily = int(round(dt2 / dt))
t_daily    = t[::step_daily]

eps = 1e-8
rng = default_rng()

# parameters (Haiti 2022)
N       = 11_637_398
mu      = 4.30e-5
sigma   = 0.5
p       = 0.10
gamma_s = 0.20
gamma_a = 0.333
alpha_s = 0.0062
omega   = 9.13e-4
kappa   = 1e6
eta_s   = 10.0
eta_a   = 1.0
delta_param = 0.3
beta_s  = 2.84e-11
beta_a  = 9.29e-8
beta_0  = 3.748e-3

# seasonal forcing baseline
A1_BASE   = 0.131
PHI1_BASE = 2.27
A2_BASE   = 0.235
PHI2_BASE = 0.983


TAU_PHASE_RANGE    = (0.0, 365.0)
A1_RANGE           = (0.118, 0.144)   
A2_RANGE           = (0.212, 0.259)   
PHASE_JITTER_DAYS  = 20.0


rate_E  = sigma + mu                  
rate_Is = gamma_s + alpha_s + mu
rate_Ia = gamma_a + mu
rate_B  = delta_param



C_s      = (p * sigma) / ((sigma + mu) * (gamma_s + alpha_s + mu))
C_a      = ((1.0 - p) * sigma) / ((sigma + mu) * (gamma_a + mu))
C_direct = beta_s * C_s + beta_a * C_a
C_env    = (N / (kappa * delta_param)) * (eta_s * C_s + eta_a * C_a)
beta_b_c = (1.0 - C_direct) / C_env
m_c      = beta_b_c / beta_0

print(f"beta_b_c = {beta_b_c:.6f}")
print(f"m_c      = {m_c:.6f}")


def jitter_phase(base_phi, period_days, jitter_days, rng_obj):
    day_shift = rng_obj.uniform(-jitter_days, jitter_days)
    return base_phi + 2.0 * np.pi * (day_shift / period_days)

def seasonal_factor(t_scalar, tau_phase, a1_sim, phi1_sim, a2_sim, phi2_sim):
    tt = t_scalar + tau_phase
    return (
        1.0
        + a1_sim * np.sin(2.0 * np.pi * tt / 365.0 + phi1_sim)
        + a2_sim * np.sin(2.0 * np.pi * tt / 182.5 + phi2_sim)
    )

def detect_transition_ramp(m0, m1):
    """RAMP rule (C4): crossing when m0 + m1*t reaches m_c. Returns (crossed, day)."""
    if m1 <= 0 or m0 >= m_c:
        return (m0 >= m_c), (0.0 if m0 >= m_c else tmax)
    t_cross = (m_c - m0) / m1
    if t_cross < tmax:
        return True, float(t_cross)
    return False, tmax

def detect_transition_Ridx(t_array, Ridx_fn):
    R_idx = np.array([Ridx_fn(tt) for tt in t_array])
    idxs = np.where(R_idx >= 1.0)[0]
    if idxs.size == 0:
        return False, tmax
    return True, float(t_array[idxs[0]])


def derivatives(S, E, Is, Ia, R, B, t_scalar, Npop, beta_b_fn, ramp_on):
    beta_b_t = beta_b_fn(t_scalar, ramp_on=ramp_on)
    lam = beta_s * Is / Npop + beta_a * Ia / Npop + beta_b_t * B / (kappa + B)
    dS  = mu * Npop + omega * R - lam * S - mu * S
    dE  = lam * S - (sigma + mu) * E
    dIs = p * sigma * E - (gamma_s + alpha_s + mu) * Is
    dIa = (1.0 - p) * sigma * E - (gamma_a + mu) * Ia
    dR  = gamma_s * Is + gamma_a * Ia - (omega + mu) * R
    dB  = eta_s * Is + eta_a * Ia - delta_param * B
    return dS, dE, dIs, dIa, dR, dB


for batch in range(1, n_batches + 1):
    print(f"\n========== Starting Batch {batch} of {n_batches} ==========\n")

    
    m0_array = rng.triangular(0, m_c / 2, m_c, sim_per_batch)
    m1_array = []

    
    for j in range(sim_per_batch // 2):
        left_slope  = 0
        right_slope = (m_c - m0_array[j]) / tmax
        mid_slope   = right_slope / 2
        slope = rng.triangular(left_slope, mid_slope, right_slope)
        m1_array.append(slope)


    for j in range(sim_per_batch // 2, sim_per_batch):
        left_slope  = (m_c - m0_array[j]) / tmax
        right_slope = 2 * (m_c - m0_array[j]) / tmax
        mid_slope   = right_slope / 2
        slope = rng.triangular(left_slope, mid_slope, right_slope)
        m1_array.append(slope)

    m1_array = np.array(m1_array)

    labels       = []                      # index rule
    tbif_array   = np.zeros(sim_per_batch) # index rule
    appended_ews = []                      # index rule
    labels_ramp       = []                      # ramp rule (1e)
    tbif_ramp_array   = np.zeros(sim_per_batch) # ramp rule (1e)
    appended_ews_ramp = []                      # ramp rule (1e)
    both_rules_rows   = []                      # per-sim comparison (1e)

    print(f"Begin SEIIRB simulations for batch {batch}\n")
    for j in range(sim_per_batch):
        if (j + 1) % 100 == 0 or j == 0:
            print(f"  [Batch {batch}] Simulation {j + 1}/{sim_per_batch} ...")

        m0 = m0_array[j]
        m1 = m1_array[j]

        
        tau_phase = rng.uniform(*TAU_PHASE_RANGE)
        a1_sim    = rng.uniform(*A1_RANGE)
        a2_sim    = rng.uniform(*A2_RANGE)
        phi1_sim  = jitter_phase(PHI1_BASE, 365.0, PHASE_JITTER_DAYS, rng)
        phi2_sim  = jitter_phase(PHI2_BASE, 182.5, PHASE_JITTER_DAYS, rng)

        
        def q_sim(t_scalar):
            return seasonal_factor(t_scalar, tau_phase, a1_sim, phi1_sim, a2_sim, phi2_sim)

        def beta_b_sim(t_scalar, ramp_on=True):
            m_t = m0 if not ramp_on else (m0 + m1 * t_scalar)
            val = m_t * beta_0 * q_sim(t_scalar)
            return max(val, 1e-6)

        def R_idx_sim(t_scalar):
            beta_b_t = (m0 + m1 * t_scalar) * beta_0 * q_sim(t_scalar)
            return beta_b_t / beta_b_c

     
        E0  = rng.uniform(200, 400)
        Is0 = rng.uniform(500, 1500)
        Ia0 = Is0 * rng.uniform(8, 10)
        R0  = 0.0
        B0  = rng.uniform(500, 5000)
        pop_factor = rng.uniform(0.98, 1.02)
        S0 = N * pop_factor - E0 - Is0 - Ia0 - R0
        S0 = max(S0, 0.0)

    
        Is_floor     = Is0 * (0.1 / 7.0)
        Is_reset_max = Is0 * (1.0 / 7.0)

        
        sigma_S  = 0.0   
        sigma_E  = rng.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_E  / 2.0)   
        sigma_Is = rng.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_Is / 2.0)   
        sigma_Ia = rng.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_Ia / 2.0)   
        sigma_R  = 0.0   
        sigma_B  = rng.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_B  / 2.0)   

        
        n_burn = int(tburn / dt)
        n_main = len(t)

        dW_S_burn  = rng.normal(0.0, sigma_S  * np.sqrt(dt), n_burn)
        dW_E_burn  = rng.normal(0.0, sigma_E  * np.sqrt(dt), n_burn)
        dW_Is_burn = rng.normal(0.0, sigma_Is * np.sqrt(dt), n_burn)
        dW_Ia_burn = rng.normal(0.0, sigma_Ia * np.sqrt(dt), n_burn)
        dW_R_burn  = rng.normal(0.0, sigma_R  * np.sqrt(dt), n_burn)
        dW_B_burn  = rng.normal(0.0, sigma_B  * np.sqrt(dt), n_burn)

        dW_S  = rng.normal(0.0, sigma_S  * np.sqrt(dt), n_main)
        dW_E  = rng.normal(0.0, sigma_E  * np.sqrt(dt), n_main)
        dW_Is = rng.normal(0.0, sigma_Is * np.sqrt(dt), n_main)
        dW_Ia = rng.normal(0.0, sigma_Ia * np.sqrt(dt), n_main)
        dW_R  = rng.normal(0.0, sigma_R  * np.sqrt(dt), n_main)
        dW_B  = rng.normal(0.0, sigma_B  * np.sqrt(dt), n_main)

        
        S_val, E_val, Is_val, Ia_val, R_val, B_val = S0, E0, Is0, Ia0, R0, B0
        for i in range(n_burn):
            dS, dE, dIs, dIa, dR, dB = derivatives(
                S_val, E_val, Is_val, Ia_val, R_val, B_val,
                0.0, N, beta_b_sim, ramp_on=False
            )
            S_val  += dS  * dt + S_val  * dW_S_burn[i]
            E_val  += dE  * dt + E_val  * dW_E_burn[i]
            Is_val += dIs * dt + Is_val * dW_Is_burn[i]
            Ia_val += dIa * dt + Ia_val * dW_Ia_burn[i]
            R_val  += dR  * dt + R_val  * dW_R_burn[i]
            B_val  += dB  * dt + B_val  * dW_B_burn[i]

            if S_val  < 0: S_val  = eps
            if E_val  < 0: E_val  = eps
            if Is_val < 0: Is_val = eps
            if Ia_val < 0: Ia_val = eps
            if R_val  < 0: R_val  = eps
            if B_val  < 0: B_val  = eps

       
        if Is_val < Is_floor: Is_val = rng.uniform(Is_floor, Is_reset_max)

        
        Is_arr = np.empty(n_main)
        Is_arr[0] = Is_val
        S_curr, E_curr, Is_curr, Ia_curr, R_curr, B_curr = S_val, E_val, Is_val, Ia_val, R_val, B_val

        for i in range(n_main - 1):
            tt = t[i]
            dS, dE, dIs, dIa, dR, dB = derivatives(
                S_curr, E_curr, Is_curr, Ia_curr, R_curr, B_curr,
                tt, N, beta_b_sim, ramp_on=True
            )
            S_next  = S_curr  + dS  * dt + S_curr  * dW_S[i]
            E_next  = E_curr  + dE  * dt + E_curr  * dW_E[i]
            Is_next = Is_curr + dIs * dt + Is_curr * dW_Is[i]
            Ia_next = Ia_curr + dIa * dt + Ia_curr * dW_Ia[i]
            R_next  = R_curr  + dR  * dt + R_curr  * dW_R[i]
            B_next  = B_curr  + dB  * dt + B_curr  * dW_B[i]

            S_curr  = max(S_next,  eps)
            E_curr  = max(E_next,  eps)
            Is_curr = Is_next if Is_next >= Is_floor else rng.uniform(Is_floor, Is_reset_max)   
            Ia_curr = max(Ia_next, eps)
            R_curr  = max(R_next,  eps)
            B_curr  = max(B_next,  eps)

            Is_arr[i + 1] = Is_curr

        
        
        outbreak, t_outbreak = detect_transition_Ridx(t, R_idx_sim)
        tbif_array[j] = t_outbreak
        labels.append(1 if outbreak else 0)

        
        outbreak_ramp, t_ramp = detect_transition_ramp(m0, m1)
        tbif_ramp_array[j] = t_ramp
        labels_ramp.append(1 if outbreak_ramp else 0)

        both_rules_rows.append({
            "sim": j + 1, "m0": m0, "m1": m1, "Is0": Is0,
            "label_index_rule": int(outbreak), "crossing_day_index_rule": t_outbreak,
            "label_ramp_rule": int(outbreak_ramp), "crossing_day_ramp_rule": t_ramp,
        })

        
        Is_daily = Is_arr[::step_daily]
        ts_series = pd.Series(Is_daily, index=pd.Index(t_daily, name="Time"))

        def ews_block(transition_day):
            ews_obj = ewstools.core.TimeSeries(ts_series, transition=transition_day)
            ews_obj.detrend(method="Lowess", span=span)
            ews_obj.compute_auto(rolling_window=rw, lag=1)
            ews_obj.compute_var(rolling_window=rw)
            df_tmp = pd.concat([ews_obj.state, ews_obj.ews], axis=1)
            df_tmp["tsid"]     = j + 1
            df_tmp["Variable"] = "Is"
            return df_tmp

        appended_ews.append(ews_block(t_outbreak))        
        appended_ews_ramp.append(ews_block(t_ramp))       

    
    def assemble(appended):
        return (pd.concat(appended).reset_index().rename(columns={"index": "Time"})
                .set_index(["tsid", "Variable", "Time"]).sort_index())
    df_ews      = assemble(appended_ews)        
    df_ews_ramp = assemble(appended_ews_ramp)   
    label_df      = pd.DataFrame(labels)
    label_df_ramp = pd.DataFrame(labels_ramp)

    n_outbreaks = int(label_df.iloc[:, 0].sum())
    n_ramp      = int(label_df_ramp.iloc[:, 0].sum())
    print(f"\n[Batch {batch}] Label balance, INDEX rule: {n_outbreaks} transition, {sim_per_batch - n_outbreaks} null")
    print(f"[Batch {batch}] Label balance, RAMP rule:  {n_ramp} transition, {sim_per_batch - n_ramp} null")

    
    print(f"[Batch {batch}] Saving CSV files ...\n")

    for rule, df_rule, lab_rule, tb_rule in (
        ("index", df_ews,      label_df,      tbif_array),
        ("ramp",  df_ews_ramp, label_df_ramp, tbif_ramp_array),
    ):
        cfg = LABEL_RULES[rule]
        for i_sim in range(sim_per_batch):
            blk = df_rule.loc[i_sim + 1, "Is"].reset_index()
            blk[["Time", "state"]].rename(columns={"state": "I"}).to_csv(
                os.path.join(cfg["sims"], f"tseries_batch{batch}_sim{i_sim+1}.csv"), index=False)
            blk[["Time", "residuals"]].to_csv(
                os.path.join(cfg["resids"], f"resids_batch{batch}_sim{i_sim+1}.csv"), index=False)
        lab_rule.to_csv(os.path.join(cfg["labels"], f"label_batch{batch}.csv"), header=False, index=False)
        pd.DataFrame(tb_rule).to_csv(os.path.join(cfg["labels"], f"outbreak_times_batch{batch}.csv"), header=False, index=False)
        print(f"[Batch {batch}] saved {rule.upper()} rule -> {cfg['root']}")

    pd.DataFrame(both_rules_rows).to_csv(
        os.path.join(both_rules_folder, f"crossing_days_both_rules_batch{batch}.csv"), index=False)

    print(f"==> Batch {batch} complete.\n")

    del df_ews, label_df, appended_ews, labels
    del df_ews_ramp, label_df_ramp, appended_ews_ramp, labels_ramp, both_rules_rows
    gc.collect()

print("\nAll batches completed successfully!")
