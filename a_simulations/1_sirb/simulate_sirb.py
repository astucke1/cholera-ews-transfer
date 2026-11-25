####################################################################
# simulate_sirb.py -- SIRB training-set generation (pipeline step 1)
#
# 20 batches x 1,000 stochastic SIRB simulations with a linearly ramping transmission rate.
# Half the ramps cross the threshold alpha_c before tmax (label 1, transition) and half do
# not (label 0, null). Each daily infected series is detrended with ewstools (Lowess) and the
# time series, residuals and labels are written to output/SIRB/. Runtime: roughly 3 hours.
#
# Model following Codeco (2001) with Haiti demographics from Andrews & Basu (2011);
# pipeline structure adapted from Bury et al. (2021) and Chakraborty et al. (2024).
#
# The SIRB model:
#   dS/dt = mu - mu*S - alpha(t)*S*(B/(kappa+B))
#   dI/dt = alpha(t)*S*(B/(kappa+B)) - (gamma+mu)*I
#   dR/dt = gamma*I - mu*R
#   dB/dt = nu*I - xi*B
#
# where alpha(t) = alpha0 + alpha1*t is the ramping transmission rate.
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

output_root = os.path.join(SCRIPT_DIR, "output", "SIRB")
output_sims_folder   = os.path.join(output_root, "output_sims")
output_resids_folder = os.path.join(output_root, "output_resids")
output_labels_folder = os.path.join(output_root, "output_labels")
for folder in [output_sims_folder, output_resids_folder, output_labels_folder]:
    os.makedirs(folder, exist_ok=True)

for folder in [output_sims_folder, output_resids_folder, output_labels_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)



n_batches = 20
sim_per_batch = 1000


dt = 0.01
t0 = 0
tmax = 1500
tburn = 100


dt2 = 1
rw = 0.25
span = 0.2

H = 1e4
mu = 1.0 / (61 * 365)          
gamma = 0.2                    
kappa = 1e6                     
nu = 10.0 * H                 
xi = 0.33                       
eps = 1e-8          



S0_init = 0.99
I0_init = 0.01
R0_init = 0.0
B0_init = 0.01


I_FLOOR     = I0_init * (0.1 / 7.0)
I_RESET_MAX = I0_init * (1.0 / 7.0)



t = np.arange(t0, tmax, dt)


def dS_dt(S, B, alpha, mu, kappa):
    return mu - alpha * S * (B / (kappa + B)) - mu * S

def dI_dt(S, B, I, alpha, mu, gamma, kappa):
    return alpha * S * (B / (kappa + B)) - (gamma + mu) * I

def dR_dt(I, R, mu, gamma):
    return gamma * I - mu * R

def dB_dt(I, B, nu, xi):
    return nu * I - xi * B



for batch in range(1, n_batches + 1):
    print(f"\n========== Starting Batch {batch} ==========\n")

    
    alpha_c = (kappa * xi * (gamma + mu)) / nu
    print(f"  alpha_c = {alpha_c:.4f}")


    
    alpha0_array = np.random.triangular(0, alpha_c / 2, alpha_c, sim_per_batch)
    alpha1_array = []

    for j in range(sim_per_batch // 2):
        left_slope  = 0
        right_slope = (alpha_c - alpha0_array[j]) / tmax
        mid_slope   = right_slope / 2
        slope = np.random.triangular(left_slope, mid_slope, right_slope)
        alpha1_array.append(slope)

    for j in range(sim_per_batch // 2, sim_per_batch):
        left_slope  = (alpha_c - alpha0_array[j]) / tmax
        right_slope = (2 * alpha_c - alpha0_array[j]) / tmax
        mid_slope   = right_slope / 2
        slope = np.random.triangular(left_slope, mid_slope, right_slope)
        alpha1_array.append(slope)

    alpha1_array = np.array(alpha1_array)

    
    list_traj_append = []
    label_list = []
    tbif_array = np.zeros(sim_per_batch)

    
    for j in range(sim_per_batch):

        if (j + 1) % 50 == 0 or j == 0:
            print(f"  [Batch {batch}] Simulation {j+1}/{sim_per_batch} ...")

        
        
       
        sigma_S = 0.0  
        sigma_I = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt((gamma + mu) / 2.0)   
        sigma_R = 0.0  
        sigma_B = np.random.triangular(0.0, 0.5, 1.0) * np.sqrt(xi / 2.0)   
        alpha0 = alpha0_array[j]
        alpha1 = alpha1_array[j]
        alpha_t = alpha0 + alpha1 * t

        
        if alpha_t[-1] >= alpha_c:
            tbif = t[np.where(alpha_t >= alpha_c)[0][0]]
        else:
            tbif = tmax
        tbif_array[j] = tbif

        
        
        n_burn = int(tburn / dt)

        dW_S_burn = np.random.normal(0, sigma_S * np.sqrt(dt), n_burn)
        dW_I_burn = np.random.normal(0, sigma_I * np.sqrt(dt), n_burn)
        dW_R_burn = np.random.normal(0, sigma_R * np.sqrt(dt), n_burn)
        dW_B_burn = np.random.normal(0, sigma_B * np.sqrt(dt), n_burn)

        dW_S = np.random.normal(0, sigma_S * np.sqrt(dt), len(t))
        dW_I = np.random.normal(0, sigma_I * np.sqrt(dt), len(t))
        dW_R = np.random.normal(0, sigma_R * np.sqrt(dt), len(t))
        dW_B = np.random.normal(0, sigma_B * np.sqrt(dt), len(t))

        
    
        S_val = S0_init
        I_val = I0_init
        R_val = R0_init
        B_val = B0_init

        for i in range(n_burn):
            alpha_current = alpha_t[0]
            S_val += dS_dt(S_val, B_val, alpha_current, mu, kappa) * dt + S_val * dW_S_burn[i]
            I_val += dI_dt(S_val, B_val, I_val, alpha_current, mu, gamma, kappa) * dt + I_val * dW_I_burn[i]
            R_val += dR_dt(I_val, R_val, mu, gamma) * dt + R_val * dW_R_burn[i]
            B_val += dB_dt(I_val, B_val, nu, xi) * dt + B_val * dW_B_burn[i]

            if S_val < 0: S_val = eps
            if I_val < 0: I_val = eps
            if R_val < 0: R_val = eps
            if B_val < 0: B_val = eps

        if I_val < I_FLOOR: I_val = np.random.uniform(I_FLOOR, I_RESET_MAX)


        
        S_arr = np.zeros(len(t))
        I_arr = np.zeros(len(t))
        R_arr = np.zeros(len(t))
        B_arr = np.zeros(len(t))

        S_arr[0] = S_val
        I_arr[0] = I_val
        R_arr[0] = R_val
        B_arr[0] = B_val

        for i in range(len(t) - 1):
            alpha_current = alpha_t[i]

            S_arr[i+1] = S_arr[i] + dS_dt(S_arr[i], B_arr[i], alpha_current, mu, kappa) * dt + S_arr[i] * dW_S[i]
            I_arr[i+1] = I_arr[i] + dI_dt(S_arr[i], B_arr[i], I_arr[i], alpha_current, mu, gamma, kappa) * dt + I_arr[i] * dW_I[i]
            R_arr[i+1] = R_arr[i] + dR_dt(I_arr[i], R_arr[i], mu, gamma) * dt + R_arr[i] * dW_R[i]
            B_arr[i+1] = B_arr[i] + dB_dt(I_arr[i], B_arr[i], nu, xi) * dt + B_arr[i] * dW_B[i]

            if S_arr[i+1] < 0: S_arr[i+1] = eps
            if I_arr[i+1] < I_FLOOR: I_arr[i+1] = np.random.uniform(I_FLOOR, I_RESET_MAX)   
            if R_arr[i+1] < 0: R_arr[i+1] = eps
            if B_arr[i+1] < 0: B_arr[i+1] = eps
        
        df_temp = pd.DataFrame({
            'tsid': (j + 1) * np.ones(len(t)),
            'Time': t,
            'S': S_arr,
            'I': I_arr,
            'R': R_arr,
            'B': B_arr
        })
        list_traj_append.append(df_temp)

        
    
        if tbif < tmax:
            label_list.append(pd.DataFrame([1]))
        else:
            label_list.append(pd.DataFrame([0]))


    df_traj = pd.concat(list_traj_append).set_index(['tsid', 'Time'])
    label = pd.concat(label_list, ignore_index=True)


    print(f"\n[Batch {batch}] Now computing EWS on 'I' variable.\n")

    df_traj_filt = df_traj.loc[::int(dt2 / dt)]
    appended_ews = []

    for i_sim in range(sim_per_batch):
        if (i_sim + 1) % 50 == 0 or i_sim == 0:
            print(f"    EWS for sim {i_sim+1}/{sim_per_batch} in batch {batch} ...")

        ts = df_traj_filt.loc[i_sim + 1]['I']
        ews_dic = ewstools.core.TimeSeries(ts, transition=tbif_array[i_sim])
        ews_dic.detrend(method='Lowess', span=span)
        ews_dic.compute_auto(rolling_window=rw, lag=1)
        ews_dic.compute_var(rolling_window=rw)

        z1 = ews_dic.state
        z2 = ews_dic.ews
        df_ews_temp = pd.concat([z1, z2], axis=1)
        df_ews_temp['tsid'] = i_sim + 1
        df_ews_temp['Variable'] = 'I'
        appended_ews.append(df_ews_temp)

    df_ews = pd.concat(appended_ews).reset_index().set_index(['tsid', 'Variable', 'Time'])

    print(f"[Batch {batch}] Saving CSV files ...\n")

    for i_sim in range(sim_per_batch):
        Ind_tseries = (
            df_ews.sort_index().loc[i_sim + 1, 'I'].reset_index()[['Time', 'state']]
        )
        tseries_rename = Ind_tseries.rename(columns={'state': 'I'})

        filepath_ts = os.path.join(
            output_sims_folder, f"tseries_batch{batch}_sim{i_sim+1}.csv")
        tseries_rename.to_csv(filepath_ts, index=False)

        df_resids = (
            df_ews.sort_index().loc[i_sim + 1, 'I'].reset_index()[['Time', 'residuals']]
        )
        filepath_resids = os.path.join(
            output_resids_folder, f"resids_batch{batch}_sim{i_sim+1}.csv")
        df_resids.to_csv(filepath_resids, index=False)

    filepath_label = os.path.join(output_labels_folder, f"label_batch{batch}.csv")
    label.to_csv(filepath_label, header=False, index=False)

    filepath_bif = os.path.join(output_labels_folder,
                                f"bifurcation_points_batch{batch}.csv")
    pd.DataFrame(tbif_array).to_csv(filepath_bif, header=False, index=False)

    print(f"==> Batch {batch} finished. Files saved.\n")

    del df_traj, df_ews, label, list_traj_append, appended_ews
    gc.collect()

print("\nAll batches completed successfully!")
