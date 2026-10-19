# Bayesian neural network with random effects (BNN-RE) for ongoing interval forecasting

Code and result summaries accompanying "Bayesian neural network with random effects for ongoing interval forecasting of nursery mortality in commercial swine production" (Zhang, Magalhães, Linhares, Wang; Frontiers in Veterinary Science, submitted).

## Files

| File | Purpose |
|---|---|
| `prep_real.py` | Data preparation for the production records (requires the proprietary workbook, not included): unit correction, one record per nursery lot (health events aggregated over the lot's barns), winsorisation at the training percentiles, min–max scaling, one-hot encoding, evaluation blocks. Prints the counts quoted in the paper. |
| `bnnre_lib.py` | Model (BNN-RE, BNN, and the linear mixed model LMM-RE when `hid = 0`), mean-field variational fitting (Pyro `AutoDiagonalNormal`, Adam, optional JIT ELBO), posterior recycling for the updates, vectorised posterior predictive sampling, interval width and coverage. |
| `config.py` | Hyperparameters of the field study (10 hidden units, learning rate 0.005, 1,500 epochs: the cross-validated values), number of predictive draws, tempering block size, methods. |
| `cv_hyper.py` | Five-fold cross-validation of the BNN-RE hyperparameters on the training records (grid in the paper); `results_cv/cv_results.csv` holds its output. |
| `run_real.py`, `run_seeds.sh` | Field study, BNN, BNN-RE and LMM-RE: initial fit and four five-week updates, consecutive and cumulative schedules, one seed per process. |
| `run_real_tempered.py`, `run_temper_seeds.sh` | Field study, BNN-RE-T: the BNN-RE update with the tempered recycled prior for the residual scale (`n_0` = 60 is the second argument of the driver). |
| `run_field_all.sh` | Runs the two field drivers for seeds 567–616. |
| `conformal.py` | BNN-RE-C: conformal-style calibration of the interval half-width on the blocks already forecast, computed after the fact from the per-observation predictions (`calibrate_sequence(blocks, alpha)`; the paper uses `alpha = 0.04`). |
| `run_sim.py` | Simulation design (constants, predictor pool, mean functions) and a per-replicate Pyro implementation used to validate the batched code. |
| `sim_batched.py`, `run_sim_all.sh` | Simulation study: all replicates of one functional form fitted at once in PyTorch (same model, variational family and objective as the Pyro code); the driver runs the three forms for the 4-update design (BNN, BNN-RE, BNN-RE-T, LMM-RE), the reverse-drift scenario (`SE_TE=0.75`) and the 20-update horizon. |
| `make_table1.py` | Table 1 from the cleaned data. |
| `make_results.py` | Every results table, figure and reported number (`key_numbers.json`) from the saved outputs; works from the released summaries alone. |
| `make_schematic.py` | Figure 1 (schematic) with matplotlib. |
| `build_all.sh` | Rebuilds tables, figures, the manuscript and this release. |
| `results_real/summary_seed*.csv`, `results_real/calib_seed*.csv` | Per-seed field results (width, coverage, MSE, posterior means of sigma_e and sigma_u) for BNN, BNN-RE and LMM-RE, and the per-seed BNN-RE-C results. |
| `results_real_tempered/summary_seed*.csv` | Per-seed field results for BNN-RE-T. |
| `results_sim/`, `results_sim_rev/`, `results_sim_long/` | Per-replicate simulation results (`sim_<form>[_rev|_K21]_rep*.csv`) and the per-replicate BNN-RE-C results (`calib_*.csv`). |
| `results_cv/cv_results.csv` | Cross-validation results. |
| `key_numbers.json` | Every number quoted in the paper, as produced by `make_results.py`. |

## Method summary

* Model: `y_ij = NN_theta(x_ij) + u_i + e_ij`, one hidden tanh layer (LMM-RE: `x_ij' w + b` instead of the network); priors `theta ~ N(0, 1)`, `u_i | sigma_u ~ N(0, sigma_u)`, `sigma_u = s^2`, `s ~ U(0.4, 0.8)`, `sigma_e ~ Beta(3, 3)` (Normal distributions are written with their standard deviation).
* Inference: mean-field Gaussian variational family on the unconstrained scale, Adam (learning rate 0.005 for the field data, 0.01 for the simulation), full-batch SVI (1,500 epochs for the field data; 1,500 / 2,000 / 2,500 for the linear / nonlinear / NN simulation forms), 5,000 posterior predictive draws, 95% equal-tailed intervals.
* BNN-RE update: the variational posterior of (theta, u, sigma_e) becomes the prior; the population variance sigma_u is not re-estimated, while each source effect is carried forward by source identity and updated through its recycled marginal. Constrained parameters receive their Gaussian on the unconstrained scale (logit for `sigma_e`). Consecutive updating recycles the preceding posterior and refits on the newest block; cumulative updating recycles the initial posterior and refits on all blocks observed so far. Each update is a full variational fit on the block(s) used, not a single gradient pass.
* BNN-RE-T: as BNN-RE, but the recycled prior of `logit(sigma_e)` has its standard deviation multiplied by `sqrt(n_prev / n_0)` with `n_0` equal to the block size (60 in the field data and the 4-update simulation, 50 in the 20-update simulation): a power prior with discount `n_0 / n_prev` on `logit(sigma_e)` only.
* BNN-RE-C: standardised scores `|y - c| / h` (centre and half-width of the BNN-RE interval) from all blocks already forecast; the half-width for the next block is multiplied by the `ceil((n + 1)(1 - alpha))`-th smallest score with `alpha = 0.04`; the first block (and any block whose calibration set would hold fewer than 50 scores) keeps the BNN-RE interval. No finite-sample coverage guarantee is claimed; coverage is evaluated empirically.
* Note: Pyro keeps the guide parameters in a global store under a fixed name, so predictions are always made immediately after the corresponding fit; recycled priors are read from the fitted guide object.

## Requirements

Python 3.11+, `torch` (CPU), `pyro-ppl` 1.9, `numpy`, `pandas`, `scipy`, `openpyxl`, `matplotlib`, `pillow`.

## Reproducing the paper

From the released summaries (no data needed; regenerates Tables 2–5, Figures 2–5 and `key_numbers.json`):
```
python make_results.py results_real results_real_tempered results_sim results_sim_long tables figures_tiff results_sim_rev
python make_schematic.py Figure_1.png Figure_1.tif
```
From the production workbook (`data.xlsx`, available from the corresponding author with the permission of the production system):
```
python prep_real.py data.xlsx                     # cleaning counts
python cv_hyper.py data.xlsx results_cv/cv_results.csv
./run_field_all.sh                                # BNN, BNN-RE, LMM-RE (50 seeds) and BNN-RE-T -> out_real, out_real_t
./run_sim_all.sh data.xlsx 100                    # simulations -> out_sim, out_sim_rev, out_sim_long
./build_all.sh
```
Runs are independent over seeds and can be split across processes. One 1,500-epoch field fit takes about 5 s on one CPU core with the JIT ELBO; the batched simulation fits 100 replicates of one form in a few minutes per update sequence. The simulation uses the predictor rows of the production data; any matrix of 67 standardised inputs can replace the pool to run it without the proprietary file.
