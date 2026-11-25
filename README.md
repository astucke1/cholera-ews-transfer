# Transferring simulation-trained early warning classifiers to a new disease: four conditions from cholera

This repository contains the code, trained models and summary results of the paper "Transferring simulation-trained early warning classifiers to a new disease: four conditions from cholera" (Stuckey and Wan).

The training data were generated from two cholera transmission models (SIRB and SIARB) and the test data from a seasonally forced SEIARB model fitted to Haiti's 2022 resurgence (called SEIIRB in the code). The deep learning training, applying and testing codes are edited from published codes by Bury et al. (2021), Deep learning for early warning signals of tipping points, PNAS, and Chakraborty et al. (2024), An early warning indicator trained on stochastic disease-spreading models with different noises.

Steps:

step-1: simulate training and test data (codes are in the "a_simulations" folder: SIRB, SIARB, SEIARB, then combine the two training sets).

step-2: train the CNN-LSTM classifiers at input lengths 100 and 500 (codes are in the "b_training" folder). The trained models are also given in this folder.

step-3: apply the trained models to the SEIARB test simulations (code is in the "c_dl_apply" folder).

step-4: compute Kendall tau for variance and lag-1 AC (code is in "d_plots/2_classical_ews").

step-5: evaluate the predictions (code is in "d_plots/1_evaluate_predictions").

step-6: generate the figures of the paper ("make_figures.py") and print the other numbers reported in the text ("reported_numbers.py"). Both run in seconds from the files in "results".

Steps 4 and 5 and the apply step take "--label-rule index" to repeat the analysis under the seasonal-index transition definition reported as a sensitivity analysis. The default is the slow-parameter definition used for the main results.

* "results" folder contains the scores, labels, and figures used in the paper. The SIDATR scores were obtained by applying the released models of Chakraborty et al. (2024) to the same test simulations with the same preprocessing.

* "e_data" folder contains the Haiti surveillance and rainfall data and the fitted SEIARB parameters.

The simulated time series and per-simulation predictions are not included because of their size.
