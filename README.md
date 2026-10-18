# Bayesian neural network with random effects (BNN-RE) for ongoing forecasting

Code and derived results accompanying "Bayesian Neural Network (BNN) with Random Effects for Ongoing Forecasting of Nursery Mortality in Commercial Swine Production" (Zhang, Magalhães, Linhares, Wang; Frontiers in Veterinary Science, submitted).

## Files

| File | Purpose |
|---|---|
| `bnnre_lib.py` | Model (BNN-RE and BNN), mean-field variational fitting (Pyro `AutoDiagonalNormal`, Adam), posterior recycling for online updates, vectorised posterior predictive sampling, interval width and coverage. |
| `prep_real.py` | Data preparation for the production records (requires the proprietary workbook, not included); reproduces the record and predictor counts reported in the paper. |
| `run_real.py` | Field study, BNN and BNN-RE: initial fit and four five-week updates, consecutive and cumulative schedules, one seed per process (`run_seeds.sh` loops over seeds). Saves per-seed summaries and the per-observation interval bounds used by the calibration. |
| `run_sim_rest.sh` | Driver for the simulation runs reported in the paper. |
| `run_real_tempered.py` | Field study, BNN-RE-T: the BNN-RE update with the tempered recycled prior for the residual scale (`run_temper_seeds.sh` loops over seeds; the block size `n_b = 60` is the third argument). |
| `conformal.py` | BNN-RE-C: conformal-style calibration of the interval half-width on the blocks already forecast, computed after the fact from the saved per-observation predictions (`calibrate_sequence(blocks, alpha)`; the paper uses `alpha = 0.04`). |
| `sim_batched.py` | Simulation study: all replicates of one functional form fitted at once in pure PyTorch (same model, variational family and objective as the Pyro code). `python sim_batched.py data.xlsx out_sim2 <linear|nonlinear|nn> 100 5` for the 5-block design; `CUMU=0 python sim_batched.py data.xlsx out_sim_long <form> 100 21` for the 20-update horizon. `VARIANTS=a,c` (default) runs BNN-RE and BNN-RE-T. Predictors are resampled from the production data; replace the predictor pool with any matrix of 67 standardised inputs to run it without the proprietary file. |
| `run_sim.py` | Per-replicate Pyro implementation of the simulation (used to validate `sim_batched.py`; the two agree to within Monte Carlo error). |
| `analyze_improvements.py` | Exploratory comparison of the refinements on the field data, including the combinations not used in the paper (calibration on the BNN, calibration on the tempered variant). |
| `make_results.py` | Builds every results table, figure and reported number in the manuscript from the saved run outputs (`key_numbers.json`). |
| `make_schematic.py` | Draws the schematic of the procedure (Figure 1) with matplotlib. |
| `results_real/summary_seed*.csv`, `results_real_tempered/summary_seed*.csv` | Per-seed field results (width, coverage, MSE, posterior means of sigma_e and sigma_u, by method, strategy and test set). |
| `results_sim/sim_*_rep*.csv`, `results_sim_long/sim_*_K21_rep*.csv` | Per-replicate simulation results (5-block and 21-block designs). |

## Method summary

* Model: `y_ij = NN_theta(x_ij) + u_i + e_ij`, one hidden tanh layer; priors `theta ~ N(0, 1)`, `u_i | sigma_u ~ N(0, sigma_u)`, `sigma_u = s^2`, `s ~ U(0.4, 0.8)`, `sigma_e ~ Beta(3, 3)`.
* Inference: mean-field Gaussian variational family on the unconstrained scale, Adam (learning rate 0.005 for the field data, 0.01 for the simulation), full-batch SVI (2,000 epochs for the field data; 1,500 / 2,000 / 2,500 for the linear / nonlinear / NN simulation forms), 5,000 posterior predictive draws, 95% equal-tailed intervals.
* BNN-RE update: the variational posterior of (theta, u, sigma_e) becomes the prior; sigma_u stays at its initial posterior. Constrained parameters receive their Gaussian on the unconstrained scale (logit for `sigma_e`). Each source effect's marginal is carried forward by source identity. Consecutive updating recycles the preceding posterior and uses the newest block; cumulative updating recycles the initial posterior and uses all blocks observed so far. (`fit()` also accepts `init=prior` to warm-start an update at the recycled prior; the reported runs use Pyro's default initialisation.)
* BNN-RE-T: as BNN-RE, but the recycled prior of `logit(sigma_e)` has its standard deviation multiplied by `sqrt(n_prev / n_0)` with `n_0 = 60` (a power prior with discount `n_0 / n_prev` on sigma_e only), so the previous posterior counts as 60 observations' worth (about five weeks of groups) of information about the residual scale.
* BNN-RE-C: standardised scores `|y - c| / h` (centre and half-width of the BNN-RE interval) from all blocks already forecast; the half-width for the next block is multiplied by the `ceil((n + 1)(1 - alpha))`-th smallest score with `alpha = 0.04`; the first block (and any block whose calibration set would hold fewer than 50 scores) keeps the BNN-RE interval.
* Note: Pyro keeps the guide parameters in a global store under a fixed name, so predictions are always made immediately after the corresponding fit; recycled priors are read from the fitted guide object.

## Requirements

Python 3.11+, `torch` (CPU), `pyro-ppl` 1.9, `numpy`, `pandas`, `scipy`, `openpyxl`, `matplotlib`, `pillow`.

## Reproducing the paper

```
./run_seeds.sh out_real_jit $(seq 567 616)       # BNN and BNN-RE, 50 seeds
./run_temper_seeds.sh out_real_t60 60 $(seq 567 616)   # BNN-RE-T
./run_sim_rest.sh                                 # simulations -> out_sim2 (5 blocks) and out_sim_long (21 blocks)
python make_results.py out_real_jit out_real_t60 out_sim2 out_sim_long tables figures_tiff
python make_schematic.py figures/Figure_1.png figures_tiff/Figure_1.tif
```
Runs are independent over seeds and can be split across processes. One 2,000-epoch field fit takes about 20 s on one CPU core (about 5 s with Pyro's JIT-compiled ELBO, which `run_real.py` uses); the batched simulation fits 100 replicates in a few minutes per form and update sequence.
