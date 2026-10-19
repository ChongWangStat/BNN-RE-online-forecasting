"""Exploratory variant: BNN-RE with a tempered recycled prior for the residual standard deviation.
At each update the recycled prior for eta_e (= logit sigma_e) keeps the location of the previous
posterior but its scale is inflated by sqrt(n_prev / n0), so that the previous posterior counts as n0
observations instead of n_prev (n0 = one block of groups).  Everything else is the BNN-RE
update of run_real.py (source effects carried forward by identity, sigma_u fixed).
Usage: python run_real_tempered.py XLSX OUTDIR N0 SEED [SEED ...]"""
import sys, os, time, math, numpy as np, pandas as pd, torch
from prep_real import load, Design
from bnnre_lib import make_model, fit, recycled_prior, predict, evaluate, posterior_scalars

from config import HID, LR, STEPS, NDRAW, JIT
xlsx, out, N0 = sys.argv[1], sys.argv[2], float(sys.argv[3]); seeds = [int(s) for s in sys.argv[4:]]
os.makedirs(out, exist_ok=True); torch.set_num_threads(1)
train, blocks, info = load(xlsx); D = Design(train)
x, y, z, _ = D.xyz(train); B = [D.xyz(b) for b in blocks]; n_src, in_dim = len(D.levels), x.shape[1]

def temper(prior, n_prev):
    loc, sc = prior['eta_e']; pr = dict(prior); pr['eta_e'] = (loc, sc * math.sqrt(n_prev / N0)); return pr

def cat(ks):
    return torch.cat([B[k][0] for k in ks]), torch.cat([B[k][1] for k in ks]), torch.cat([B[k][2] for k in ks])

for seed in seeds:
    if os.path.exists(f'{out}/summary_seed{seed}.csv'): continue
    t0 = time.time(); rows = []; perobs = {}
    def log(strategy, k_eval, guide, model):
        xe, ye, ze, _ = B[k_eval]; ev = evaluate(predict(model, guide, xe, ze, seed, NDRAW), ye)
        perobs[f'{strategy}|{k_eval + 1}'] = np.stack([ev['lo'], ev['hi'], ev['mean'], ev['y']])
        rows.append(dict(seed=seed, method=f'BNN-RE-T{int(N0)}', strategy=strategy, eval_block=k_eval + 1, width=ev['width'], cover=ev['cover'], mse=ev['mse'], **posterior_scalars(guide, False)))
        return rows[-1]
    m0 = make_model(in_dim, HID, n_src, True); g0, _ = fit(m0, x, z, y, seed, LR, STEPS, jit=JIT)
    log('initial', 0, g0, m0)
    prev_g, n_prev = g0, x.shape[0]
    for k in range(1, 5):
        pr = temper(recycled_prior(prev_g), n_prev); mk = make_model(in_dim, HID, n_src, True, prior=pr)
        xb, yb, zb, _ = B[k - 1]; gk, _ = fit(mk, xb, zb, yb, seed, LR, STEPS, jit=JIT)
        row = log('cons', k, gk, mk)
        if k == 1: upd1 = dict(row, strategy='cumu'); perobs['cumu|2'] = perobs['cons|2']
        prev_g, n_prev = gk, N0 + xb.shape[0]
    rows.append(upd1)
    for k in range(2, 5):
        pr = temper(recycled_prior(g0), x.shape[0]); mk = make_model(in_dim, HID, n_src, True, prior=pr)
        xc, yc, zc = cat(range(k)); gk, _ = fit(mk, xc, zc, yc, seed, LR, STEPS, jit=JIT)
        log('cumu', k, gk, mk)
    pd.DataFrame(rows).to_csv(f'{out}/summary_seed{seed}.csv', index=False); np.savez_compressed(f'{out}/perobs_seed{seed}.npz', **perobs)
    print(f'[{seed}] tempered n0={N0:.0f} done in {time.time() - t0:.0f}s', flush=True)
