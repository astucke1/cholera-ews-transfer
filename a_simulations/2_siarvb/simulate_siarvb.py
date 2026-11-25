####################################################################
# simulate_siarvb.py -- SIARVB training-set generation (pipeline step 2)
#
# 20 batches x 1,000 stochastic SIARVB simulations with a linearly ramping transmission rate,
# labelled, detrended and written to output/SIARVB/ exactly as in simulate_sirb.py.
# Runtime: roughly 7 hours.
#
# Model from Andrews & Basu (2011), adapted from Bury et al. (2021).
#
# The SIARVB model:
#   dS/dt  = mu*N + omega*R + epsilon*V - alpha*S*(BL/(kL+BL) + BH/(kH+BH)) - mu*S
#   dI/dt  = (1-p)*alpha*S*(BL/(kL+BL) + BH/(kH+BH)) - (mu_C + mu + (1-theta)*gamma + theta*gamma*lambd)*I
#   dA/dt  = p*alpha*S*(BL/(kL+BL) + BH/(kH+BH)) - (mu + gamma)*A
#   dR/dt  = ((1-theta)*gamma + theta*gamma*lambd)*I + gamma*A - (mu + omega)*R
#   dV/dt  = -(epsilon + mu)*V
#   dBH/dt = ((psi*theta + (1-theta))*xi_S*I + xi_A*A)/W - chi*BH
#   dBL/dt = chi*BH - delta*BL
#
# where alpha(t) = alpha0 + alpha1*t is the ramping control parameter.
#
####################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ewstools
import random
import sys
import math
import gc


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR   = os.path.join(REPO_ROOT, "e_data")

output_root = os.path.join(SCRIPT_DIR, "output", "SIARVB")
output_sims_folder   = os.path.join(output_root, "output_sims")
output_resids_folder = os.path.join(output_root, "output_resids")
output_labels_folder = os.path.join(output_root, "output_labels")
for folder in [output_sims_folder, output_resids_folder, output_labels_folder]:
    os.makedirs(folder, exist_ok=True)


n_batches = 20
sim_per_batch = 1000


dt = 0.01
t0 = 0
tmax = 1500
tburn = 100
dt2 = 1
rw = 0.25
span = 0.2
lags = [1]
ews = ['var','ac']

t = np.arange(t0, tmax, dt)

eps = 1e-8


# parameters (Andrews & Basu 2011)

mu = 1.0 / (61 * 365)
omega = 1.0 / (0.8 * 365)
epsilon = 1.0 / (2 * 365)
mu_C = 0.0247
gamma = 1.0 / 5.0
p = 0.79
kappa_L = 1e5
kappa_H = kappa_L / 50.0
xi_S = 1.3e11
xi_A = 1.3e8
theta = 0.08
psi = 0.52
lambd = 2.3
chi = 1.0
delta = 1.0 / 30.0
tau = 0.67
W = 15.0 * 365.0

# alpha_c from next-generation matrix R0 = 1
alpha_c = 6.37071e-05

# IC
I0 = 0.00046
A0 = 0.0
S0 = 1.0 - I0
R0 = 0.0
V0 = 0.0
BH0 = 0.001
BL0 = 0.001

# C1: reintroduction floor on I, following Chakraborty et al. (2024).
# their released code does:  if I < 0.1: I = random.uniform(0.1, 1)
# in a 500-person SIR with I0 = 7. carried over here as the same
# proportion of I0:  threshold = (0.1/7)*I0,  reset ~ uniform(threshold, I0/7)
I_FLOOR     = I0 * (0.1 / 7.0)
I_RESET_MAX = I0 * (1.0 / 7.0)

# C3: noise intensity matched to Chakraborty et al. (2024) through the stationary variance of the
# linearised noise, sigma^2 / (2 r). They draw sigma_C ~ triangular(0, 0.5, 1) against an I removal
# rate of 2 per day; each compartment X here gets sigma_X = sigma_C * sqrt(r_X / 2), with r_X its
# own removal rate. C2: S, R (and V) carry no noise.
rate_I  = mu_C + mu + (1 - theta)*gamma + theta*gamma*lambd   # removal rate of I
rate_A  = mu + gamma                                           # removal rate of A
rate_BH = chi
rate_BL = delta


# ode

def dS_dt(S, R, V, alpha, BL, BH):
    infection_term = alpha * S * ((BL / (kappa_L + BL)) + (BH / (kappa_H + BH)))
    return mu + omega * R + epsilon * V - infection_term - mu * S

def dI_dt(S, alpha, BL, BH, I):
    infection_gain = (1 - p) * alpha * S * ((BL / (kappa_L + BL)) + (BH / (kappa_H + BH)))
    removal = (mu_C + mu + (1 - theta)*gamma + theta*gamma*lambd) * I
    return infection_gain - removal

def dA_dt(S, alpha, BL, BH, A):
    infection_gain = p * alpha * S * ((BL / (kappa_L + BL)) + (BH / (kappa_H + BH)))
    removal = (mu + gamma) * A
    return infection_gain - removal

def dR_dt(I, A, R):
    recovery = ((1 - theta)*gamma + theta*gamma*lambd) * I + gamma * A
    removal = (mu + omega) * R
    return recovery - removal

def dV_dt(V):
    return -(epsilon + mu) * V

def dBH_dt(I, A, BH):
    return ((psi*theta + (1 - theta))*xi_S*I + xi_A*A) / W - chi * BH

def dBL_dt(BH, BL):
    return chi * BH - delta * BL


for batch in range(1, n_batches + 1):
    print(f"\n========== Starting Batch {batch} of {n_batches} ==========\n")

    alpha0_array = np.random.triangular(0, alpha_c/2, alpha_c, sim_per_batch)
    alpha1_array = []

    for j in range(sim_per_batch // 2):
        left_slope = 0
        right_slope = (alpha_c - alpha0_array[j]) / tmax
        mid_slope = right_slope / 2
        slope = np.random.triangular(left_slope, mid_slope, right_slope)
        alpha1_array.append(slope)

    for j in range(sim_per_batch // 2, sim_per_batch):
        left_slope = (alpha_c - alpha0_array[j]) / tmax
        right_slope = 2 * (alpha_c - alpha0_array[j]) / tmax
        mid_slope = right_slope / 2
        slope = np.random.triangular(left_slope, mid_slope, right_slope)
        alpha1_array.append(slope)

    alpha1_array = np.array(alpha1_array)

    list_traj_append = []
    label_list = []
    tbif_array = np.zeros(sim_per_batch)

    print(f"Begin simulations for batch {batch}\n")
    for j in range(sim_per_batch):
        if (j+1) % 100 == 0 or j == 0:
            print(f"  [Batch {batch}] Simulation {j+1}/{sim_per_batch} ...")

        # noise intensities
        sigma_S  = 0.0   # C2: no noise (no restoring force at mu ~ 1/61 yr)
        sigma_I  = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_I  / 2.0)   # C3
        sigma_A  = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_A  / 2.0)   # C3
        sigma_R  = 0.0   # C2: no noise (no restoring force at mu ~ 1/61 yr)
        sigma_V  = 0.0   # C2: no noise (no restoring force at mu ~ 1/61 yr)
        sigma_BH = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_BH / 2.0)   # C3
        sigma_BL = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt(rate_BL / 2.0)   # C3

      
        alpha0 = alpha0_array[j]
        alpha1 = alpha1_array[j]
        alpha_t = alpha0 + alpha1 * t


        if alpha_t[-1] >= alpha_c:
            tbif = t[np.where(alpha_t >= alpha_c)[0][0]]
        else:
            tbif = tmax
        tbif_array[j] = tbif

 
        n_burn = int(tburn / dt)
        dW_S_burn  = np.random.normal(0, sigma_S  * np.sqrt(dt), n_burn)
        dW_I_burn  = np.random.normal(0, sigma_I  * np.sqrt(dt), n_burn)
        dW_A_burn  = np.random.normal(0, sigma_A  * np.sqrt(dt), n_burn)
        dW_R_burn  = np.random.normal(0, sigma_R  * np.sqrt(dt), n_burn)
        dW_V_burn  = np.random.normal(0, sigma_V  * np.sqrt(dt), n_burn)
        dW_BH_burn = np.random.normal(0, sigma_BH * np.sqrt(dt), n_burn)
        dW_BL_burn = np.random.normal(0, sigma_BL * np.sqrt(dt), n_burn)

        dW_S  = np.random.normal(0, sigma_S  * np.sqrt(dt), len(t))
        dW_I  = np.random.normal(0, sigma_I  * np.sqrt(dt), len(t))
        dW_A  = np.random.normal(0, sigma_A  * np.sqrt(dt), len(t))
        dW_R  = np.random.normal(0, sigma_R  * np.sqrt(dt), len(t))
        dW_V  = np.random.normal(0, sigma_V  * np.sqrt(dt), len(t))
        dW_BH = np.random.normal(0, sigma_BH * np.sqrt(dt), len(t))
        dW_BL = np.random.normal(0, sigma_BL * np.sqrt(dt), len(t))

     
        S_val  = S0
        I_val  = I0
        A_val  = A0
        R_val  = R0
        V_val  = V0
        BH_val = BH0
        BL_val = BL0

        for i in range(n_burn):
            alpha_current = alpha_t[0]
            S_val  += dS_dt(S_val, R_val, V_val, alpha_current, BL_val, BH_val)*dt + S_val*dW_S_burn[i]
            I_val  += dI_dt(S_val, alpha_current, BL_val, BH_val, I_val)*dt + I_val*dW_I_burn[i]
            A_val  += dA_dt(S_val, alpha_current, BL_val, BH_val, A_val)*dt + A_val*dW_A_burn[i]
            R_val  += dR_dt(I_val, A_val, R_val)*dt + R_val*dW_R_burn[i]
            V_val  += dV_dt(V_val)*dt + V_val*dW_V_burn[i]
            BH_val += dBH_dt(I_val, A_val, BH_val)*dt + BH_val*dW_BH_burn[i]
            BL_val += dBL_dt(BH_val, BL_val)*dt + BL_val*dW_BL_burn[i]

            if S_val  < 0: S_val  = eps
            if I_val  < 0: I_val  = eps
            if A_val  < 0: A_val  = eps
            if R_val  < 0: R_val  = eps
            if V_val  < 0: V_val  = eps
            if BH_val < 0: BH_val = eps
            if BL_val < 0: BL_val = eps

        # C1: the floor is applied once after burn-in as well, as in Chakraborty et al. (2024)
        if I_val < I_FLOOR: I_val = np.random.uniform(I_FLOOR, I_RESET_MAX)

       
        S_arr  = np.zeros(len(t))
        I_arr  = np.zeros(len(t))
        A_arr  = np.zeros(len(t))
        R_arr  = np.zeros(len(t))
        V_arr  = np.zeros(len(t))
        BH_arr = np.zeros(len(t))
        BL_arr = np.zeros(len(t))

        S_arr[0]  = S_val
        I_arr[0]  = I_val
        A_arr[0]  = A_val
        R_arr[0]  = R_val
        V_arr[0]  = V_val
        BH_arr[0] = BH_val
        BL_arr[0] = BL_val

        for i in range(len(t)-1):
            alpha_current = alpha_t[i]

            S_arr[i+1]  = S_arr[i]  + dS_dt(S_arr[i], R_arr[i], V_arr[i], alpha_current, BL_arr[i], BH_arr[i])*dt \
                                     + S_arr[i]*dW_S[i]
            I_arr[i+1]  = I_arr[i]  + dI_dt(S_arr[i], alpha_current, BL_arr[i], BH_arr[i], I_arr[i])*dt \
                                     + I_arr[i]*dW_I[i]
            A_arr[i+1]  = A_arr[i]  + dA_dt(S_arr[i], alpha_current, BL_arr[i], BH_arr[i], A_arr[i])*dt \
                                     + A_arr[i]*dW_A[i]
            R_arr[i+1]  = R_arr[i]  + dR_dt(I_arr[i], A_arr[i], R_arr[i])*dt \
                                     + R_arr[i]*dW_R[i]
            V_arr[i+1]  = V_arr[i]  + dV_dt(V_arr[i])*dt + V_arr[i]*dW_V[i]
            BH_arr[i+1] = BH_arr[i] + dBH_dt(I_arr[i], A_arr[i], BH_arr[i])*dt \
                                     + BH_arr[i]*dW_BH[i]
            BL_arr[i+1] = BL_arr[i] + dBL_dt(BH_arr[i], BL_arr[i])*dt \
                                     + BL_arr[i]*dW_BL[i]

            if S_arr[i+1]  < 0: S_arr[i+1]  = eps
            if I_arr[i+1]  < I_FLOOR: I_arr[i+1]  = np.random.uniform(I_FLOOR, I_RESET_MAX)   # C1 reintroduction floor
            if A_arr[i+1]  < 0: A_arr[i+1]  = eps
            if R_arr[i+1]  < 0: R_arr[i+1]  = eps
            if V_arr[i+1]  < 0: V_arr[i+1]  = eps
            if BH_arr[i+1] < 0: BH_arr[i+1] = eps
            if BL_arr[i+1] < 0: BL_arr[i+1] = eps

        df_temp = pd.DataFrame({
            'tsid': (j+1)*np.ones(len(t)),
            'Time': t,
            'S': S_arr,
            'I': I_arr,
            'A': A_arr,
            'R': R_arr,
            'V': V_arr,
            'BH': BH_arr,
            'BL': BL_arr
        })
        list_traj_append.append(df_temp)

        if tbif < tmax:
            label_list.append(pd.DataFrame([1]))
        else:
            label_list.append(pd.DataFrame([0]))

    df_traj = pd.concat(list_traj_append).set_index(['tsid','Time'])
    label_df = pd.concat(label_list, ignore_index=True)

    
    print(f"\n[Batch {batch}] Begin EWS computation\n")

    df_traj_filt = df_traj.loc[::int(dt2/dt)]
    appended_ews = []
    for i_sim in range(sim_per_batch):
        if (i_sim+1) % 100 == 0 or i_sim == 0:
            print(f"    EWS for sim {i_sim+1}/{sim_per_batch} in batch {batch} ...")

        ts = df_traj_filt.loc[i_sim+1]['I']
        ews_dic = ewstools.core.TimeSeries(ts, transition=tbif_array[i_sim])
        ews_dic.detrend(method='Lowess', span=span)
        ews_dic.compute_auto(rolling_window=rw, lag=1)
        ews_dic.compute_var(rolling_window=rw)

        z1 = ews_dic.state
        z2 = ews_dic.ews
        df_ews_temp = pd.concat([z1, z2], axis=1)
        df_ews_temp['tsid'] = i_sim+1
        df_ews_temp['Variable'] = 'I'
        appended_ews.append(df_ews_temp)

    df_ews = pd.concat(appended_ews).reset_index().set_index(['tsid','Variable','Time'])


    print(f"[Batch {batch}] Saving CSV files ...\n")

    for i_sim in range(sim_per_batch):
        Ind_tseries = df_ews.sort_index().loc[i_sim+1, 'I'].reset_index()[['Time','state']]
        tseries_rename = Ind_tseries.rename(columns={'state': 'I'})
        filepath_ts = os.path.join(output_sims_folder, f"tseries_batch{batch}_sim{i_sim+1}.csv")
        tseries_rename.to_csv(filepath_ts, index=False)

        df_resids = df_ews.sort_index().loc[i_sim+1, 'I'].reset_index()[['Time','residuals']]
        filepath_resids = os.path.join(output_resids_folder, f"resids_batch{batch}_sim{i_sim+1}.csv")
        df_resids.to_csv(filepath_resids, index=False)

    label_file = os.path.join(output_labels_folder, f"label_batch{batch}.csv")
    label_df.to_csv(label_file, header=False, index=False)

    bif_file = os.path.join(output_labels_folder, f"bifurcation_points_batch{batch}.csv")
    pd.DataFrame(tbif_array).to_csv(bif_file, header=False, index=False)

    print(f"==> Finished batch {batch}. Data saved.\n")

    del df_traj, df_ews, label_df, list_traj_append, appended_ews
    gc.collect()

print("\nAll batches completed successfully!")
