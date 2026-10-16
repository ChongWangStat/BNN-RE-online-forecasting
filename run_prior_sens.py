"""Prior sensitivity: initial BNN-RE and cumulative-carry updates under bounded vs half-normal scale priors. Usage: run_prior_sens.py <seed> <outdir>"""
import sys, time, numpy as np, pandas as pd, torch
from prep_data import real_data_split, Scaler, xy
from bnnre_lib import *
torch.set_num_threads(1)
seed = int(sys.argv[1]); OUT = sys.argv[2]; import os; os.makedirs(OUT, exist_ok=True)
train, test, pred_cols = real_data_split()
sc = Scaler(torch.tensor(train[pred_cols].values.astype(np.float32)))
x_tr, y_tr = xy(train, pred_cols); x_tr = sc(x_tr); levels = sorted(train.SowSource.unique())
bounds = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]; T = [test[(test.week_num > lo) & (test.week_num <= hi)] for lo, hi in bounds]
def tens(df): xx, yy = xy(df, pred_cols); return sc(xx), yy
rows = []
for variant in ['bounded', 'halfnormal']:
    spec = Spec(x_tr.shape[1], 20, levels, [], use_re=True, scale_prior=variant)
    model, guide, loss = fit(spec, x_tr, zmat(train.SowSource.tolist(), levels), y_tr, seed, 0.005, 2000)
    def ev(spec, guide, df, k, strat):
        xx, yy = tens(df); draws, _ = predict_draws(posterior_samples(guide, spec, 5000, seed), spec, xx, zmat(df.SowSource.tolist(), spec.levels))
        sc_ = score(draws, yy); post = posterior_samples(guide, spec, 2000, seed)
        rows.append(dict(seed=seed, prior=variant, step=k, strategy=strat, width=sc_['width'].mean(), cover=sc_['cover'].mean(), iscore=sc_['iscore'].mean(), crps=sc_['crps'].mean(),
                         sigma_e=float(post['sigma'].mean()), sigma_u=float((post['s'] ** 2).mean())))
    ev(spec, guide, T[0], 0, 'initial')
    init_spec, init_guide = spec, guide
    for k in range(1, 5):
        upd = pd.concat(T[:k]); sp = recycled_spec(init_spec, init_guide, upd.SowSource.tolist(), reset_re=False)
        xx, yy = tens(upd); m, g, l = fit(sp, xx, zmat(upd.SowSource.tolist(), sp.levels), yy, seed + 1000 * k, 0.005, 2000)
        ev(sp, g, T[k], k, 'cumu_a')
    print(f'[{seed}] {variant} done', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/prior_sens_seed{seed}.csv', index=False)
