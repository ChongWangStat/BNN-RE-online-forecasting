"""Held-out sow source evaluation (C3). Usage: run_newsource.py <seeds> <steps> <outdir>
Ten sow sources with >= 25 training groups are removed from training.  Their groups (all years, time order) form a stream
split into 5 chunks; after updating with chunks 1..k the model predicts chunk k+1.  For each prediction we record how many
groups of that source have already entered the update data (0, 1-4, 5-9, 10-19, 20+)."""
import sys, time, numpy as np, pandas as pd, torch
from prep_data import load_raw, design, Scaler, xy
from bnnre_lib import *
import bnnre_lib; bnnre_lib.JIT = True
torch.set_num_threads(1)
seeds = [int(s) for s in sys.argv[1].split(',')]; STEPS = int(sys.argv[2]); OUT = sys.argv[3]
import os; os.makedirs(OUT, exist_ok=True)
dat = load_raw(); comb = design(dat)
pred_cols = [c for c in comb.columns if c not in ('SowSource', 'avg_mortality_60days', 'year_num')]
counts = comb[comb.year_num != 2023].groupby('SowSource').size()
elig = sorted(counts[counts >= 25].index)
rng = np.random.default_rng(2026); held = sorted(rng.choice(elig, 10, replace=False))
print('held-out sources:', held, flush=True)
train = comb[(comb.year_num != 2023) & (~comb.SowSource.isin(held))]
stream = comb[comb.SowSource.isin(held)].sort_values(['year_num', 'week_num'])
chunks = np.array_split(np.arange(len(stream)), 5); T = [stream.iloc[c] for c in chunks]
print('train', len(train), train.SowSource.nunique(), 'stream', len(stream), [len(t) for t in T], flush=True)
sc = Scaler(torch.tensor(train[pred_cols].values.astype(np.float32)))
def tens(df): xx, yy = xy(df, pred_cols); return sc(xx), yy
x_tr, y_tr = tens(train); levels = sorted(train.SowSource.unique())
HID, LR = 20, 0.005

def evaluate(spec, guide, df, seed, seen_counts):
    xx, yy = tens(df); zz = zmat(df.SowSource.tolist(), spec.levels) if spec.use_re else torch.zeros(len(df), 0)
    draws, _ = predict_draws(posterior_samples(guide, spec, 5000, seed), spec, xx, zz)
    s = score(draws, yy); s['source'] = df.SowSource.values; s['n_seen'] = np.array([seen_counts.get(src, 0) for src in df.SowSource])
    return s

def run_seed(seed):
    t0 = time.time(); rows = []
    def log(method, step, strat, scd):
        for i in range(len(scd['y'])):
            rows.append(dict(seed=seed, method=method, step=step, strategy=strat, source=scd['source'][i], n_seen=int(scd['n_seen'][i]),
                             width=scd['width'][i], cover=scd['cover'][i], iscore=scd['iscore'][i], crps=scd['crps'][i], sqerr=scd['sqerr'][i]))
    inits = {}
    for method, hid, use_re in [('BNN-RE', HID, True), ('BNN', HID, False), ('LMM-RE', 0, True)]:
        spec = Spec(x_tr.shape[1], hid, levels if use_re else [], [], use_re=use_re)
        zz = zmat(train.SowSource.tolist(), spec.levels) if use_re else torch.zeros(len(train), 0)
        model, guide, loss = fit(spec, x_tr, zz, y_tr, seed, LR, STEPS); inits[method] = (spec, guide)
        log(method, 0, 'initial', evaluate(spec, guide, T[0], seed, {}))
    for method in ['BNN-RE', 'LMM-RE', 'BNN']:
        init_spec, init_guide = inits[method]
        strategies = [('cons', 'a'), ('cons', 'b'), ('cumu', 'a'), ('cumu', 'b')] if init_spec.use_re else [('cons', '-'), ('cumu', '-')]
        for mode, variant in strategies:
            prev_spec, prev_guide = init_spec, init_guide
            for k in range(1, 5):
                upd_df = T[k - 1] if mode == 'cons' else pd.concat(T[:k])
                base_spec, base_guide = (prev_spec, prev_guide) if mode == 'cons' else (init_spec, init_guide)
                spec = recycled_spec(base_spec, base_guide, upd_df.SowSource.tolist(), reset_re=(variant == 'b'))
                xx, yy = tens(upd_df); zz = zmat(upd_df.SowSource.tolist(), spec.levels) if spec.use_re else torch.zeros(len(upd_df), 0)
                model, guide, loss = fit(spec, xx, zz, yy, seed + 1000 * k, LR, STEPS)
                seen_counts = pd.concat(T[:k]).SowSource.value_counts().to_dict()
                log(method, k, f'{mode}_{variant}', evaluate(spec, guide, T[k], seed, seen_counts))
                prev_spec, prev_guide = spec, guide
    pd.DataFrame(rows).to_csv(f'{OUT}/newsource_seed{seed}.csv', index=False)
    print(f'[{seed}] done in {time.time()-t0:.0f}s', flush=True)
for sd in seeds: run_seed(sd)
