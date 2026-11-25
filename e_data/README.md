# e_data

Source data behind the SEIARB (SEIIRB in the code) parameterisation. The simulation scripts do not read
these files; the fitted values are hard-coded in `a_simulations/3_seiirb/simulate_seiirb.py`. The case
counts and fitted parameters are read by `make_figures.py` for the figure of the fit.

- `choleraHaiti202220232024.csv` -- weekly suspected cholera case counts, Haiti, 2022-2024.
  Source: PAHO/WHO Cholera Outbreak in Hispaniola situation reports
  (https://www.paho.org/en/cholera-outbreak-haiti-2022-situation-report). Last accessed 19 March 2025.
- `haitiRainfall20102013.csv` -- Haiti precipitation, 10-day intervals, 2010-2013.
  Source: Humanitarian Data Exchange (HDX/OCHA) (https://data.humdata.org/dataset/hti-rainfall-subnational).
  Last accessed 17 March 2025.
- `seiirb_fitted_params.json` -- fixed, fitted and derived SEIARB parameters (52-week fitting window),
  including the threshold `beta_b_c` and the ramp threshold `mc` used by the slow-parameter label rule.
