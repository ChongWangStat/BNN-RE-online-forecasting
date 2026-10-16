import sys, time, json, numpy as np, pandas as pd, torch
from prep_data import real_data_split, Scaler, xy
from bnnre_lib import *
torch.set_num_threads(1)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
seeds = [int(s) for s in sys.argv[1].split(',')]
OUT = sys.argv[3] if len(sys.argv) > 3 else 'results_real'
import os; os.makedirs(OUT, exist_ok=True)

train, test, pred_cols = real_data_split()
sc = Scaler(torch.tensor(train[pred_cols].values.astype(np.float32)))
x_tr, y_tr = xy(train, pred_cols); x_tr = sc(x_tr)
train_levels = sorted(train.SowSource.unique())
bounds = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 99)]
T = [test[(test.week_num > lo) & (test.week_num <= hi)] for lo, hi in bounds]
for k, t in enumerate(T): print('test set', k + 1, len(t), t.SowSource.nunique())
def tens(df): xx, yy = xy(df, pred_cols); return sc(xx), yy
HID, LR = 20, 0.005

def evaluate(spec, guide, df, seed):
    xx, yy = tens(df); zz = zmat(df.SowSource.tolist(), spec.levels) if spec.use_re else torch.zeros(len(df), 0)
    draws, _ = predict_draws(posterior_samples(guide, spec, 5000, seed), spec, xx, zz)
    sc_ = score(draws, yy); sc_['source'] = df.SowSource.values; sc_['week'] = df.week_num.values
    return sc_

def run_seed(seed):
    t0 = time.time(); rows = []; perobs = {}
    def log(method, step, strat, scd, extra=None):
        r = dict(seed=seed, method=method, step=step, strategy=strat, n=len(scd['y']), width=scd['width'].mean(), cover=scd['cover'].mean(),
                 iscore=scd['iscore'].mean(), crps=scd['crps'].mean(), mse=scd['sqerr'].mean())
        if extra: r.update(extra)
        rows.append(r); perobs[(method, step, strat)] = scd
    # ---------- initial models ----------
    inits = {}
    for method, hid, use_re in [('BNN-RE', HID, True), ('BNN', HID, False), ('LMM-RE', 0, True)]:
        spec = Spec(x_tr.shape[1], hid, train_levels if use_re else [], [], use_re=use_re)
        z_tr = zmat(train.SowSource.tolist(), spec.levels) if use_re else torch.zeros(len(train), 0)
        model, guide, loss = fit(spec, x_tr, z_tr, y_tr, seed, LR, STEPS)
        inits[method] = (spec, guide)
        scd = evaluate(spec, guide, T[0], seed)
        post = posterior_samples(guide, spec, 4000, seed)
        extra = dict(loss=float(loss), sigma_e=float(post['sigma'].mean()))
        if use_re:
            extra['sigma_u'] = float((post['s'] ** 2).mean())
            if method == 'BNN-RE':
                re_mean = post['re_seen'].mean(0).numpy(); re_sd = post['re_seen'].std(0).numpy()
                pd.DataFrame({'source': train_levels, 're_mean': re_mean, 're_sd': re_sd}).to_csv(f'{OUT}/re_post_seed{seed}.csv', index=False)
        log(method, 0, 'initial', scd, extra)
        print(f'  [{seed}] {method} initial: width {scd["width"].mean():.3f} cover {scd["cover"].mean():.3f} loss {loss:.3f} ({time.time()-t0:.0f}s)', flush=True)
    # ---------- online updates ----------
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
                scd = evaluate(spec, guide, T[k], seed)
                post = posterior_samples(guide, spec, 2000, seed)
                extra = dict(loss=float(loss), sigma_e=float(post['sigma'].mean()))
                if spec.use_re: extra['sigma_u'] = float((post['s'] ** 2).mean())
                log(method, k, f'{mode}_{variant}', scd, extra)
                prev_spec, prev_guide = spec, guide
            print(f'  [{seed}] {method} {mode}_{variant} done ({time.time()-t0:.0f}s)', flush=True)
    # ---------- batch-retrained BNN-RE ----------
    for k in range(1, 5):
        df = pd.concat([train] + T[:k]); levels = sorted(df.SowSource.unique())
        spec = Spec(x_tr.shape[1], HID, levels, [], use_re=True)
        xx, yy = tens(df); zz = zmat(df.SowSource.tolist(), levels)
        model, guide, loss = fit(spec, xx, zz, yy, seed + 7000 * k, LR, STEPS)
        scd = evaluate(spec, guide, T[k], seed)
        log('BNN-RE-batch', k, 'batch', scd, dict(loss=float(loss)))
    print(f'  [{seed}] batch done ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/summary_seed{seed}.csv', index=False)
    np.savez_compressed(f'{OUT}/perobs_seed{seed}.npz', **{f'{m}|{s}|{st}|{key}': v for (m, s, st), d in perobs.items() for key, v in d.items()})
    print(f'[{seed}] finished in {time.time()-t0:.0f}s', flush=True)

for sd in seeds:
    run_seed(sd)
