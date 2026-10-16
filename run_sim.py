"""Simulation study with corrected code. Usage: run_sim.py <dgp> <scenario> <seeds> <steps> <outdir>
dgp in {linear, nonlinear, nn}; scenario in {shift, noshift}."""
import sys, time, numpy as np, pandas as pd, torch
from prep_data import load_raw, design
from bnnre_lib import *
import bnnre_lib; bnnre_lib.JIT = True
torch.set_num_threads(1)
dgp, scen = sys.argv[1], sys.argv[2]
seeds = [int(s) for s in sys.argv[3].split(',')]; STEPS = int(sys.argv[4]); OUT = sys.argv[5]
import os; os.makedirs(OUT, exist_ok=True)

# predictor pool: standardised numerical predictors + dummies, exactly as in the notebooks (year dropped)
dat = load_raw().drop(columns=['year_num']); comb = design(dat)
pred_cols = [c for c in comb.columns if c not in ('SowSource', 'avg_mortality_60days')]
X = comb[pred_cols].astype(float)
num_cols = [c for c in pred_cols if X[c].nunique() > 2]
X[num_cols] = (X[num_cols] - X[num_cols].mean()) / X[num_cols].std()
Y = comb['avg_mortality_60days'].values
# linear DGP: OLS of log mortality on all standardised predictors (no intercept, as in the notebook)
beta_ols, *_ = np.linalg.lstsq(X.values, Y, rcond=None)
def f_linear(df): return df.values @ beta_ols
def f_nonlinear(df):
    x1, x2, x3, x4, x5 = (df[c].values for c in ['Born_alive_avg3', 'Pre_weaning_mortality', 'avg_parity_at_farrow3', 'TimeToFill', 'StockingDensity1'])
    return 10 * x2 / (1 + x1 ** 2) + 5 * np.sin(x3 * x4) + 2 * x5
def f_nn(df):
    x0, x1, x2, x3 = (df[c].values for c in ['Pre_weaning_mortality', 'avg_parity_at_farrow3', 'TimeToFill', 'StockingDensity1'])
    return x0 + 2 * np.tanh(x1 + 2 * x2) + 2 * x3
F = {'linear': f_linear, 'nonlinear': f_nonlinear, 'nn': f_nn}[dgp]
def trans(v, m=1.16, s=0.39):   # alpha-calibration: fixed part standardised to mean 1.16, SD 0.39
    return (v - v.mean()) / v.std() * s + m
K_TR, N_PER, K_TE, N_TE_PER = 110, 24, 50, 6
SU_TR, SE_TR = 0.28, 0.62
SU_TE, SE_TE = (0.40, 0.50) if scen == 'shift' else (0.28, 0.62)
HID, LR = 5, 0.01

def gen(seed):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(K_TR)                                   # shared directions
    idx_tr = rng.choice(len(X), K_TR * N_PER, replace=True)
    pool = np.setdiff1d(np.arange(len(X)), np.unique(idx_tr))
    idx_te = rng.choice(pool, K_TE * N_TE_PER, replace=False)
    Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
    ftr, fte = trans(F(Xtr)), trans(F(Xte))
    subj_tr = np.repeat(np.arange(K_TR), N_PER); subj_te = np.repeat(np.arange(K_TE), N_TE_PER)   # test subjects = first 50 training subjects
    ytr = ftr + SU_TR * r[subj_tr] + rng.normal(0, SE_TR, len(ftr))
    yte = fte + SU_TE * r[subj_te] + rng.normal(0, SE_TE, len(fte))
    perm = rng.permutation(len(yte))
    Xte, yte, subj_te = Xte.iloc[perm], yte[perm], subj_te[perm]
    to_t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    train = dict(x=to_t(Xtr.values), y=to_t(ytr), src=[f's{i}' for i in subj_tr])
    tests = []
    for k in range(5):
        sl = slice(60 * k, 60 * (k + 1))
        tests.append(dict(x=to_t(Xte.values[sl]), y=to_t(yte[sl]), src=[f's{i}' for i in subj_te[sl]]))
    return train, tests

def cat(ds): return dict(x=torch.cat([d['x'] for d in ds]), y=torch.cat([d['y'] for d in ds]), src=sum([d['src'] for d in ds], []))

def evaluate(spec, guide, d, seed):
    zz = zmat(d['src'], spec.levels) if spec.use_re else torch.zeros(len(d['y']), 0)
    draws, _ = predict_draws(posterior_samples(guide, spec, 5000, seed), spec, d['x'], zz)
    return score(draws, d['y'])

def run_seed(seed):
    t0 = time.time(); train, T = gen(seed); rows = []
    def log(method, step, strat, scd, extra=None):
        r = dict(seed=seed, dgp=dgp, scenario=scen, method=method, step=step, strategy=strat, width=scd['width'].mean(), cover=scd['cover'].mean(),
                 iscore=scd['iscore'].mean(), crps=scd['crps'].mean(), mse=scd['sqerr'].mean()); r.update(extra or {}); rows.append(r)
    levels = sorted(set(train['src']), key=lambda s: int(s[1:]))
    inits = {}
    for method, hid, use_re in [('BNN-RE', HID, True), ('BNN', HID, False), ('LMM-RE', 0, True)]:
        spec = Spec(train['x'].shape[1], hid, levels if use_re else [], [], use_re=use_re)
        zz = zmat(train['src'], spec.levels) if use_re else torch.zeros(len(train['y']), 0)
        model, guide, loss = fit(spec, train['x'], zz, train['y'], seed, LR, STEPS); inits[method] = (spec, guide)
        post = posterior_samples(guide, spec, 2000, seed)
        extra = dict(loss=float(loss), sigma_e=float(post['sigma'].mean())); 
        if use_re: extra['sigma_u'] = float((post['s'] ** 2).mean())
        log(method, 0, 'initial', evaluate(spec, guide, T[0], seed), extra)
    for method in ['BNN-RE', 'LMM-RE', 'BNN']:
        init_spec, init_guide = inits[method]
        strategies = [('cons', 'a'), ('cons', 'b'), ('cumu', 'a'), ('cumu', 'b')] if init_spec.use_re else [('cons', '-'), ('cumu', '-')]
        for mode, variant in strategies:
            prev_spec, prev_guide = init_spec, init_guide
            for k in range(1, 5):
                upd = T[k - 1] if mode == 'cons' else cat(T[:k])
                base_spec, base_guide = (prev_spec, prev_guide) if mode == 'cons' else (init_spec, init_guide)
                spec = recycled_spec(base_spec, base_guide, upd['src'], reset_re=(variant == 'b'))
                zz = zmat(upd['src'], spec.levels) if spec.use_re else torch.zeros(len(upd['y']), 0)
                model, guide, loss = fit(spec, upd['x'], zz, upd['y'], seed + 1000 * k, LR, STEPS)
                post = posterior_samples(guide, spec, 2000, seed)
                extra = dict(loss=float(loss), sigma_e=float(post['sigma'].mean()))
                if spec.use_re: extra['sigma_u'] = float((post['s'] ** 2).mean())
                log(method, k, f'{mode}_{variant}', evaluate(spec, guide, T[k], seed), extra)
                prev_spec, prev_guide = spec, guide
    for k in range(1, 5):   # batch-retrained BNN-RE
        d = cat([train] + T[:k]); lv = sorted(set(d['src']), key=lambda s: int(s[1:]))
        spec = Spec(d['x'].shape[1], HID, lv, [], use_re=True)
        model, guide, loss = fit(spec, d['x'], zmat(d['src'], lv), d['y'], seed + 7000 * k, LR, STEPS)
        log('BNN-RE-batch', k, 'batch', evaluate(spec, guide, T[k], seed), dict(loss=float(loss)))
    pd.DataFrame(rows).to_csv(f'{OUT}/sim_{dgp}_{scen}_seed{seed}.csv', index=False)
    print(f'[{dgp} {scen} {seed}] done in {time.time()-t0:.0f}s', flush=True)

for sd in seeds: run_seed(sd)
