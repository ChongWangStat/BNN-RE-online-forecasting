"""Simulation study (corrected implementation).  Usage: python run_sim.py XLSX OUTDIR FORM REP [REP ...]
FORM in {linear, nonlinear, nn}.  Design follows the original notebooks (0509*.ipynb):
110 training subjects x 24 obs, 50 test subjects x 6 obs split into 5 blocks of 60 after a random
permutation; sigma_u 0.28 -> 0.40 and sigma_e 0.62 -> 0.50 from training to test; predictors
resampled from the production data (standardised numerical fields, 0/1 dummies); 100 replicates.
Corrections relative to the notebooks: (i) test subjects are the first 50 training subjects and use
their own standard-normal draw r_i, with design matrices indexed by subject id in both periods;
(ii) the affine scaling of the mean function (target mean 1.16, SD 0.39) is determined on the
training set and applied unchanged to the test set.  Hyperparameters per form as in the notebooks:
hidden units 5, learning rate 0.01, epochs 1500 (linear), 2000 (nonlinear), 2500 (nn).  The ELBO is
evaluated with Pyro's JIT-compiled Trace_ELBO for speed (same objective)."""
import sys, os, time, numpy as np, pandas as pd, torch
from bnnre_lib import make_model, fit, recycled_prior, predict, evaluate, posterior_scalars

xlsx, out, form = sys.argv[1], sys.argv[2], sys.argv[3]; reps = [int(r) for r in sys.argv[4:]]
os.makedirs(out, exist_ok=True); torch.set_num_threads(1)
HID, LR, NDRAW = 5, 0.01, 5000
STEPS = dict(linear=1500, nonlinear=2000, nn=2500)[form]
K_TR, N_PER, K_TE, N_TE = 110, 24, 50, 6
SU_TR, SE_TR, SU_TE, SE_TE = 0.28, 0.62, 0.40, 0.50
T_MEAN, T_SD = 1.16, 0.39

# ---- predictor pool from the production data (sim-notebook preprocessing) ----
na = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
dat = pd.read_excel(xlsx, keep_default_na=False, na_values=na)
dat['SowFarmMed'] = dat['SowFarmMed'].replace('None', 'none'); dat['WeanPigShot'] = dat['WeanPigShot'].replace('None', 'none')
dat = dat.dropna().sort_values(by=['SowSource', 'year_num', 'week_num', 'CloseDate'])
dat.drop(columns=['lot_no2', 'lot_nur', 'CloseDate', 'LotPrefix', 'HeadIn', 'mortality_60days', 'ADG', 'year_num'], inplace=True)
dat = dat.drop_duplicates()
num = dat.select_dtypes(include=np.number); numz = (num - num.mean()) / num.std()
cat = dat.select_dtypes(include=['object', 'string']); cat = pd.get_dummies(cat[[c for c in cat.columns if c != 'SowSource']]).astype(int)
comb = pd.concat([numz, cat], axis=1)
all_x = comb[[c for c in comb if c != 'avg_mortality_60days']]; all_y = comb['avg_mortality_60days']
assert all_x.shape[1] == 67, all_x.shape
beta = np.linalg.lstsq(all_x.values.astype(float), all_y.values.astype(float), rcond=None)[0]   # linear form (OLS, no intercept, as in the notebook)

def f_mean(X):
    if form == 'linear':
        return X.values.astype(float) @ beta
    if form == 'nonlinear':   # 10 x2/(1+x1^2) + 5 sin(x3 x4) + 2 x5
        x1, x2, x3, x4, x5 = (X[c].values.astype(float) for c in ['Born_alive_avg3', 'Pre_weaning_mortality', 'avg_parity_at_farrow3', 'TimeToFill', 'StockingDensity1'])
        return 10 * x2 / (1 + x1 ** 2) + 5 * np.sin(x3 * x4) + 2 * x5
    if form == 'nn':          # x0 + 2 tanh(x1 + 2 x2) + 2 x3
        x0, x1, x2, x3 = (X[c].values.astype(float) for c in ['Pre_weaning_mortality', 'avg_parity_at_farrow3', 'TimeToFill', 'StockingDensity1'])
        return x0 + 2 * np.tanh(x1 + 2 * x2) + 2 * x3
    raise ValueError(form)

def gen(i):
    rng = np.random.default_rng(40 + i)
    r = rng.standard_normal(K_TR)                                        # shared subject directions
    Xtr = all_x.sample(n=K_TR * N_PER, replace=True, random_state=40 + i)
    Xte = all_x.drop(Xtr.index.unique()).sample(n=K_TE * N_TE, random_state=40 + i)
    ftr, fte = f_mean(Xtr), f_mean(Xte)
    a = T_SD / ftr.std(); b = T_MEAN - a * ftr.mean()                    # scaling fixed on the training set
    ftr, fte = a * ftr + b, a * fte + b
    subj_tr = np.repeat(np.arange(K_TR), N_PER); subj_te = np.repeat(np.arange(K_TE), N_TE)
    ytr = ftr + SU_TR * r[subj_tr] + rng.normal(0, SE_TR, len(ftr))
    yte = fte + SU_TE * r[subj_te] + rng.normal(0, SE_TE, len(fte))
    perm = rng.permutation(len(yte)); Xte, yte, subj_te = Xte.iloc[perm], yte[perm], subj_te[perm]
    def Z(s):
        M = torch.zeros(len(s), K_TR); M[torch.arange(len(s)), torch.tensor(s)] = 1.0; return M
    T = lambda A: torch.tensor(np.asarray(A, dtype=np.float32))
    train = (T(Xtr.values), T(ytr), Z(subj_tr))
    blocks = [(T(Xte.values[60 * k:60 * (k + 1)]), T(yte[60 * k:60 * (k + 1)]), Z(subj_te[60 * k:60 * (k + 1)])) for k in range(5)]
    return train, blocks

def run_rep(i):
    t0 = time.time(); rows = []
    (x, y, z), B = gen(i); in_dim = x.shape[1]
    def cat_(ks):
        return torch.cat([B[k][0] for k in ks]), torch.cat([B[k][1] for k in ks]), torch.cat([B[k][2] for k in ks])
    def log(method, strategy, k_eval, guide, model):
        xe, ye, ze = B[k_eval]
        ev = evaluate(predict(model, guide, xe, ze, 1000 + i, NDRAW), ye)
        rows.append(dict(form=form, rep=i, method=method, strategy=strategy, eval_block=k_eval + 1,
                         width=ev['width'], cover=ev['cover'], mse=ev['mse'], **posterior_scalars(guide, False)))
        return rows[-1]
    for method, use_re in (('BNN', False), ('BNN-RE', True)):
        m0 = make_model(in_dim, HID, K_TR, use_re); g0, _ = fit(m0, x, z, y, 1000 + i, LR, STEPS, jit=True)
        log(method, 'initial', 0, g0, m0)
        for var, reset in ([('a', False), ('b', True)] if use_re else [('-', False)]):
            name = method if method == 'BNN' else f'BNN-RE({var})'
            prev = g0; upd1 = None
            for k in range(1, 5):
                mk = make_model(in_dim, HID, K_TR, use_re, prior=recycled_prior(prev), reset_re=reset)
                gk, _ = fit(mk, B[k - 1][0], B[k - 1][2], B[k - 1][1], 1000 + i, LR, STEPS, jit=True)
                row = log(name, 'cons', k, gk, mk); prev = gk
                if k == 1: upd1 = dict(row, strategy='cumu')   # Update #1 is the same fit under both schedules
            rows.append(upd1)
            for k in range(2, 5):
                mk = make_model(in_dim, HID, K_TR, use_re, prior=recycled_prior(g0), reset_re=reset)
                xc, yc, zc = cat_(range(k)); gk, _ = fit(mk, xc, zc, yc, 1000 + i, LR, STEPS, jit=True)
                log(name, 'cumu', k, gk, mk)
    pd.DataFrame(rows).to_csv(f'{out}/sim_{form}_rep{i}.csv', index=False)
    print(f'[{form} {i}] {time.time() - t0:.0f}s', flush=True)

for i in reps:
    if os.path.exists(f'{out}/sim_{form}_rep{i}.csv'): continue
    run_rep(i)
