"""Field study: initial fit + four sequential updates, BNN, BNN-RE and the linear mixed model comparator (LMM-RE),
consecutive and cumulative schedules, one seed per call.  Usage: python run_real.py XLSX OUTDIR SEED [SEED ...]
Hyperparameters (HID, LR, STEPS) are the cross-validated values (cv_hyper.py); see config.py."""
import sys, os, time, json, numpy as np, pandas as pd, torch
from prep_real import load, Design
from bnnre_lib import make_model, fit, recycled_prior, predict, evaluate, posterior_scalars
from config import HID, LR, STEPS, NDRAW, JIT, METHODS_REAL

xlsx, out = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3:]]
os.makedirs(out, exist_ok=True)
torch.set_num_threads(1)
train, blocks, info = load(xlsx)
D = Design(train)
x, y, z, _ = D.xyz(train)
B = [D.xyz(b) for b in blocks]                       # (x, y, z, sources) per block
n_src, in_dim = len(D.levels), x.shape[1]
assert in_dim == info['n_inputs'] and n_src == info['n_sources_train']

def cat(ks):
    xs, ys, zs = zip(*[(B[k][0], B[k][1], B[k][2]) for k in ks])
    return torch.cat(xs), torch.cat(ys), torch.cat(zs)

def run_seed(seed):
    t0 = time.time(); rows = []; perobs = {}
    def log(method, strategy, k_eval, guide, model, scal):
        xe, ye, ze, _ = B[k_eval]
        ev = evaluate(predict(model, guide, xe, ze, seed, NDRAW), ye)
        rows.append(dict(seed=seed, method=method, strategy=strategy, eval_block=k_eval + 1,
                         width=ev['width'], cover=ev['cover'], mse=ev['mse'], **scal))
        perobs[f'{method}|{strategy}|{k_eval + 1}'] = np.stack([ev['lo'], ev['hi'], ev['mean'], ev['y']])
        return rows[-1]
    for method, use_re, hid in METHODS_REAL:          # ('BNN', False, HID), ('BNN-RE', True, HID), ('LMM-RE', True, 0)
        m0 = make_model(in_dim, hid, n_src, use_re)
        g0, loss0 = fit(m0, x, z, y, seed, LR, STEPS, jit=JIT)
        log(method, 'initial', 0, g0, m0, posterior_scalars(g0, True))
        name = method
        # consecutive: recycle the preceding posterior, fit on the newest block only
        prev_g = g0
        upd1 = None
        for k in range(1, 5):
            pr = recycled_prior(prev_g); mk = make_model(in_dim, hid, n_src, use_re, prior=pr)
            xb, yb, zb, _ = B[k - 1]
            gk, _ = fit(mk, xb, zb, yb, seed, LR, STEPS, jit=JIT)
            row = log(name, 'cons', k, gk, mk, posterior_scalars(gk, False))
            if k == 1:   # Update #1 is the same fit under both schedules; re-label rather than re-predict
                upd1 = dict(row, strategy='cumu'); perobs[f'{name}|cumu|2'] = perobs[f'{name}|cons|2']
            prev_g = gk
        # cumulative: recycle the initial posterior, fit on all blocks so far
        rows.append(upd1)
        for k in range(2, 5):
            pr = recycled_prior(g0); mk = make_model(in_dim, hid, n_src, use_re, prior=pr)
            xc, yc, zc = cat(range(k))
            gk, _ = fit(mk, xc, zc, yc, seed, LR, STEPS, jit=JIT)
            log(name, 'cumu', k, gk, mk, posterior_scalars(gk, False))
        print(f'[{seed}] {method} done {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(f'{out}/summary_seed{seed}.csv', index=False)
    np.savez_compressed(f'{out}/perobs_seed{seed}.npz', **perobs)
    print(f'[{seed}] finished in {time.time() - t0:.0f}s', flush=True)

for s in seeds:
    if os.path.exists(f'{out}/summary_seed{s}.csv'):
        print(f'[{s}] exists, skipping'); continue
    run_seed(s)
