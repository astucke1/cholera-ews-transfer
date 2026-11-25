# results

Summary results used in the paper. Running the pipeline regenerates everything here. The full simulation outputs (tens of GB) are not included.

`<rule>` is `ramp` (slow-parameter rule, main results) or `index` (seasonal-index rule, sensitivity analysis). `<W>` is the input length, 100 or 500.

- `scores/`: the classifier scores behind every AUC in the paper. One row per simulation and revelation step, last five steps only (100,000 rows). Columns: `batch`, `sim`, `step` (5 = final), `y_true`, `p`.
  - `cholera_trained_<rule>_W<W>.csv`: average of networks 1_1 and 1_2.
  - `sidatr_<rule>_W<W>.csv`: the released classifiers of Chakraborty et al. (2024), applied to the same simulations.
- `classical_ews/<rule>_rule/`: lag-1 autocorrelation and variance scores (cumulative Kendall tau, last five values).
- `labels/<rule>_rule/`: label (0/1) and crossing day of each of the 20,000 test simulations.
- `labels/both_rules/`: for each simulation, the ramp intercept and slope, and the label and crossing day under both rules.
- `training/`: test-set results of each trained network on the SIRB and SIARB simulations.
- `figures/`: the figures of the paper and the values plotted, from `make_figures.py`.
