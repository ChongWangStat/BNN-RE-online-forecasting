# BNN-RE: online Bayesian neural networks with random effects for forecasting nursery mortality

Code and derived results for

> Zhang D, Magalhães ES, Linhares DCL, Wang C. Bayesian Neural Network (BNN) with Random Effects for Ongoing Forecasting of Nursery Mortality in Commercial Swine Production. Submitted to *Frontiers in Veterinary Science* (Veterinary Epidemiology and Economics), 2026.

The method fits a Bayesian neural network with a sow-farm-source random intercept by mean-field variational inference and updates it online by recycling the variational posterior as the prior each time a block of groups closes out. The same posterior predictive draws give a 95% prediction interval for each group of pigs and the probability that the group exceeds a mortality action threshold.

## What is in this repository

| Path | Content |
|---|---|
| `bnnre_lib.py` | Models (BNN-RE, BNN, LMM-RE), variational fitting (Pyro `AutoDiagonalNormal`), posterior recycling for online updates with the push-forward of constrained parameters, predictive sampling, scoring (coverage, width, interval score, CRPS, exceedance probability). |
| `prep_data.py` | Data preparation (variable selection, exclusions, one-hot encoding, standardisation, train/test split). Requires the production records, which are **not** distributed. |
| `run_real.py` | Field-data study: initial models, four online update strategies, BNN and LMM-RE comparators, batch-retrained BNN-RE. |
| `run_newsource.py` | Evaluation for sow farm sources held out of training. |
| `run_sim.py` | Simulation study with the linear, nonlinear and neural-network mean functions, with and without a shift in the variance components. |
| `run_prior_sens.py` | Prior sensitivity analysis (half-normal scale priors). |
| `aggregate_*.py` | Build the tables, figures and the numbers quoted in the manuscript from the per-seed result files. |
| `verify_bug.py` | Reproduces the diagnosis of the unconstrained-scale recycling in the earlier implementation (Supplementary Note S1 of the manuscript). |
| `results_real/` | Per-seed block-level results of the field-data study (`summary_seed*.csv`) and posterior means of the sow farm random effects (`re_post_seed*.csv`). |
| `results_newsource/` | Per-seed results of the held-out-source evaluation. |
| `results_sim/` | Per-seed results of the simulation study (synthetic data). |
| `results_prior/` | Per-seed results of the prior sensitivity analysis. |
| `outputs/tables/`, `outputs/figures/` | Aggregated tables (CSV and LaTeX), the manuscript figures (PDF and PNG) and the `numbers_*.tex` macro files used by the manuscript. |

## Data

The production records (groups of pigs from one U.S. Midwestern swine production system, 2019 to 2023) are proprietary to the participating system and are not included. They are available from the corresponding author on reasonable request and with the permission of the system. Per-observation model outputs are withheld for the same reason; the per-seed block-level summaries and all aggregated results are included.

`prep_data.py` expects the data file at `data/COMPLETE_DATA_MAR_2024.xlsx` or at the path in the environment variable `BNNRE_DATA`.

## Model summary

* `y_ij = NN_theta(x_ij) + u_i + e_ij`, one hidden tanh layer. Priors: `theta ~ N(0, 1)`, `u_i | sigma_u ~ N(0, sigma_u)`, `sigma_u = s^2` with `s ~ U(0.4, 0.8)`, `sigma_e ~ Beta(3, 3)`. Normal distributions are parameterised by mean and standard deviation.
* Inference: mean-field Gaussian on the unconstrained scale (Pyro `AutoDiagonalNormal`), Adam, 2,000 full-batch iterations (learning rate 0.005 for the field data, 0.01 for the simulation), 5,000 posterior predictive draws.
* Online update: the variational posterior becomes the prior. Constrained parameters receive the push-forward of their Gaussian through the bijection (logit-normal for `sigma_e`, scaled logit-normal for `s`). Strategy "carry": random effects of known sources are recycled and new sources receive `N(0, sigma_u)`. Strategy "re-learn": all random effects are re-drawn from `N(0, sigma_u)` with `sigma_u` recycled. Each strategy runs as a consecutive update (previous posterior, newest block) or a cumulative update (initial posterior, all blocks).

## Requirements

Python 3.11 or later and the packages in `requirements.txt` (`torch`, `pyro-ppl` 1.9, `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`).

```
pip install -r requirements.txt
```

## Reproducing the results

With the data file in place:

```
python run_real.py 567,568,...,616 2000 results_real
python run_newsource.py 101,...,120 2000 results_newsource
for s in shift noshift; do for d in nn linear nonlinear; do python run_sim.py $d $s 1,...,50 2000 results_sim; done; done
for seed in 567 ... 576; do python run_prior_sens.py $seed results_prior; done
python aggregate_real.py; python aggregate_newsource.py; python aggregate_sim.py; python aggregate_extra.py
```

Runs are embarrassingly parallel over seeds. One 2,000-iteration fit takes about 15 s on one CPU core (about 6 s with the JIT ELBO used in `run_sim.py` and `run_newsource.py`). The aggregation scripts write to `outputs/` (or to the directory in `BNNRE_OUT`). Without the data file, `aggregate_sim.py` and the block-level parts of `aggregate_real.py` and `aggregate_newsource.py` run from the included result files; the per-observation figures (forecasts for individual sources, calibration) need the withheld per-observation outputs.

The simulation resamples predictor rows from the field data. To run it without the proprietary file, replace `load_raw()` in `run_sim.py` with any predictor matrix of the same column structure.

## License

MIT License (see `LICENSE`).
