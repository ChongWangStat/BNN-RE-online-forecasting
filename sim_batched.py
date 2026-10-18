"""Simulation study, batched implementation.

Fits all R replicates of a simulation form at once with mean-field Gaussian variational inference
written directly in PyTorch: the latent vector of every replicate has its own variational location
and scale, the single-sample reparameterised ELBO is summed over replicates (they share no parameters,
so this equals R independent optimisations), and Adam is applied to the variational parameters.
Model, priors, transforms, update rules (BNN-RE, BNN-RE(sigma_u) variant, BNN; consecutive and
cumulative), hyperparameters and predictive sampling are those of bnnre_lib.py / run_sim.py; the
only difference is the batching, which removes the per-replicate Python overhead.

Usage: python sim_batched.py XLSX OUTDIR FORM [n_reps]
"""
import sys, os, time, math, numpy as np, pandas as pd, torch
torch.set_num_threads(2)
xlsx, out, form = sys.argv[1], sys.argv[2], sys.argv[3]; R = int(sys.argv[4]) if len(sys.argv) > 4 else 100
KB = int(sys.argv[5]) if len(sys.argv) > 5 else 5      # number of evaluation blocks: 5 (original design) or 20 (long horizon)
os.makedirs(out, exist_ok=True)
sys.argv = ['run_sim.py', xlsx, out, form]            # reuse the predictor pool, mean functions and constants of run_sim.py
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_sim.py')).read().split('def gen')[0]
exec(src)                                              # defines all_x, f_mean(), HID, LR, STEPS, K_TR, N_PER, K_TE, SU_*, SE_*, T_MEAN, T_SD
torch.set_num_threads(2)
N_TE, BLK = (6, 60) if KB == 5 else (KB, 50)          # 50 test subjects x N_TE obs, split into KB blocks of BLK
N0 = BLK                                               # tempered prior: previous posterior counts as one block
TAG = '' if KB == 5 else f'_K{KB}'

def gen(i):
    rng = np.random.default_rng(40 + i)
    r = rng.standard_normal(K_TR)                                        # shared subject directions
    Xtr = all_x.sample(n=K_TR * N_PER, replace=True, random_state=40 + i)
    pool = all_x.drop(Xtr.index.unique())
    Xte = pool.sample(n=K_TE * N_TE, replace=len(pool) < K_TE * N_TE, random_state=40 + i)
    ftr, fte = f_mean(Xtr), f_mean(Xte)
    a = T_SD / ftr.std(); b = T_MEAN - a * ftr.mean()                    # scaling fixed on the training set
    ftr, fte = a * ftr + b, a * fte + b
    subj_tr = np.repeat(np.arange(K_TR), N_PER); subj_te = np.repeat(np.arange(K_TE), N_TE)
    ytr = ftr + SU_TR * r[subj_tr] + rng.normal(0, SE_TR, len(ftr))
    yte = fte + SU_TE * r[subj_te] + rng.normal(0, SE_TE, len(fte))
    perm = rng.permutation(len(yte)); Xte, yte, subj_te = Xte.iloc[perm], yte[perm], subj_te[perm]
    def Z(sj):
        M = torch.zeros(len(sj), K_TR); M[torch.arange(len(sj)), torch.tensor(sj)] = 1.0; return M
    T = lambda A: torch.tensor(np.asarray(A, dtype=np.float32))
    train = (T(Xtr.values), T(ytr), Z(subj_tr))
    blocks = [(T(Xte.values[BLK * k:BLK * (k + 1)]), T(yte[BLK * k:BLK * (k + 1)]), Z(subj_te[BLK * k:BLK * (k + 1)])) for k in range(KB)]
    return train, blocks
S_LO, S_HI, W_SCALE = 0.4, 0.8, 1.0
INIT_SCALE = 0.1
LOG2PI = math.log(2 * math.pi)

# ---------------------------------------------------------------- data: stack the replicates
t0 = time.time()
Xtr, ytr, Str, Xte, yte, Ste = [], [], [], [], [], []
for i in range(R):
    (x, y, z), B = gen(i)
    Xtr.append(x); ytr.append(y); Str.append(z.argmax(1))
    Xte.append(torch.stack([b[0] for b in B])); yte.append(torch.stack([b[1] for b in B])); Ste.append(torch.stack([b[2].argmax(1) for b in B]))
Xtr, ytr, Str = torch.stack(Xtr), torch.stack(ytr), torch.stack(Str)          # [R, n, p], [R, n], [R, n]
Xte, yte, Ste = torch.stack(Xte), torch.stack(yte), torch.stack(Ste)          # [R, 5, 60, p], [R, 5, 60], [R, 5, 60]
P = Xtr.shape[2]
print(f'[{form}] data for {R} replicates in {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- latent layout
def layout(hid, use_re, has_s):
    sizes = [('w1', hid * P), ('b1', hid), ('w2', hid), ('b2', 1)]
    if use_re: sizes.append(('re', K_TR))
    if has_s: sizes.append(('eta_s', 1))
    sizes.append(('eta_e', 1))
    idx, o = {}, 0
    for k, n in sizes: idx[k] = (o, o + n); o += n
    return idx, o

def sl(theta, idx, k): a, b = idx[k]; return theta[..., a:b]

def mean_fn(theta, idx, X, subj, hid, use_re):
    """theta [..., R, D]; X [R, n, P]; subj [R, n] -> mu [..., R, n]"""
    w1 = sl(theta, idx, 'w1').reshape(*theta.shape[:-1], hid, P)
    b1, w2, b2 = sl(theta, idx, 'b1'), sl(theta, idx, 'w2'), sl(theta, idx, 'b2')
    h = torch.tanh(torch.matmul(X, w1.transpose(-1, -2)) + b1.unsqueeze(-2))          # [..., R, n, hid]
    mu = (h * w2.unsqueeze(-2)).sum(-1) + b2                                             # [..., R, n]
    if use_re:
        re = sl(theta, idx, 're')                                                        # [..., R, K]
        sub = subj.expand(*re.shape[:-1], -1) if re.dim() > 2 else subj
        mu = mu + torch.gather(re, -1, sub)
    return mu

def sigma_of(theta, idx): return torch.sigmoid(sl(theta, idx, 'eta_e'))                 # [..., R, 1]
def s_of(theta, idx): return S_LO + (S_HI - S_LO) * torch.sigmoid(sl(theta, idx, 'eta_s'))

def log_normal(x, loc, scale):
    scale = torch.as_tensor(scale, dtype=x.dtype)
    return -0.5 * LOG2PI - torch.log(scale) - 0.5 * ((x - loc) / scale) ** 2

def log_prior(theta, idx, prior, use_re, reset_re):
    """Sum of log prior densities on the unconstrained scale. prior=None -> initial model."""
    lp = torch.zeros(theta.shape[:-1])
    if prior is None:
        for k in ('w1', 'b1', 'w2', 'b2'):
            lp = lp + log_normal(sl(theta, idx, k), 0., W_SCALE).sum(-1)
        if use_re:
            eta_s = sl(theta, idx, 'eta_s'); sg = torch.sigmoid(eta_s)
            lp = lp + (torch.log(sg) + torch.log(1 - sg)).sum(-1)                        # Uniform(s) on (0.4, 0.8), Jacobian of the scaled logit (constant dropped)
            s = s_of(theta, idx)
            lp = lp + log_normal(sl(theta, idx, 're'), 0., s ** 2).sum(-1)
        eta_e = sl(theta, idx, 'eta_e'); se = torch.sigmoid(eta_e)
        lp = lp + (2 * torch.log(se) + 2 * torch.log(1 - se) + torch.log(se) + torch.log(1 - se)).sum(-1)   # Beta(3,3) density on sigma + logit Jacobian (constant dropped)
    else:
        for k in ('w1', 'b1', 'w2', 'b2', 'eta_e'):
            lp = lp + log_normal(sl(theta, idx, k), *prior[k]).sum(-1)
        if use_re:
            if reset_re:
                lp = lp + log_normal(sl(theta, idx, 'eta_s'), *prior['eta_s']).sum(-1)
                s = s_of(theta, idx)
                lp = lp + log_normal(sl(theta, idx, 're'), 0., s ** 2).sum(-1)
            else:
                lp = lp + log_normal(sl(theta, idx, 're'), *prior['re']).sum(-1)
    return lp

def fit(X, y, subj, hid, use_re, prior, reset_re, steps, lr, seed):
    """Mean-field VI for all replicates at once. Returns (loc, scale) [R, D] and the layout."""
    torch.manual_seed(seed)
    has_s = use_re and (prior is None or reset_re)
    idx, D = layout(hid, use_re, has_s)
    Rr = X.shape[0]
    loc = torch.zeros(Rr, D); log_scale = torch.full((Rr, D), math.log(INIT_SCALE))
    if prior is not None:   # start the optimisation at the recycled prior location (as Pyro's init_to_median does, in effect)
        for k in idx:
            if k == 're' and reset_re: continue        # variant (b): source effects are re-drawn from N(0, s^2); start them at 0 as Pyro's init does
            if k in prior: a, b = idx[k]; loc[:, a:b] = prior[k][0]
    loc.requires_grad_(True); log_scale.requires_grad_(True)
    opt = torch.optim.Adam([loc, log_scale], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        scale = torch.exp(log_scale)
        theta = loc + scale * torch.randn_like(loc)
        mu = mean_fn(theta, idx, X, subj, hid, use_re)
        sigma = sigma_of(theta, idx)
        loglik = log_normal(y, mu, sigma).sum(-1)
        lq = log_normal(theta, loc, scale).sum(-1)
        elbo = loglik + log_prior(theta, idx, prior, use_re, reset_re) - lq
        loss = -elbo.sum()
        loss.backward(); opt.step()
    return loc.detach(), torch.exp(log_scale.detach()), idx

def recycled(loc, scale, idx, temper_n=None):
    pr = {k: (loc[:, a:b], scale[:, a:b]) for k, (a, b) in idx.items()}
    if temper_n is not None:   # tempered prior for sigma_e: the previous posterior (fitted on temper_n obs) counts as N0 obs
        l, sc = pr['eta_e']; pr['eta_e'] = (l, sc * math.sqrt(temper_n / N0))
    return pr

def predict_eval(loc, scale, idx, hid, use_re, X, y, subj, seed, M=5000, chunk=250, keep=None, key=None):
    """95% interval width, coverage and MSE per replicate from M posterior predictive draws."""
    torch.manual_seed(seed)
    draws = []
    for c in range(0, M, chunk):
        m = min(chunk, M - c)
        theta = loc + scale * torch.randn(m, *loc.shape)                                # [m, R, D]
        mu = mean_fn(theta, idx, X, subj, hid, use_re)                                   # [m, R, n]
        sigma = sigma_of(theta, idx)                                                     # [m, R, 1]
        draws.append(mu + sigma * torch.randn_like(mu))
    d = torch.cat(draws).numpy()                                                         # [M, R, n]
    lo, hi = np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)
    yv = y.numpy()
    if keep is not None and key is not None: keep[key] = np.stack([lo, hi, d.mean(0), yv], 1).astype(np.float32)   # [R, 4, n]
    return dict(width=(hi - lo).mean(1), cover=((lo < yv) & (yv < hi)).mean(1), mse=((d.mean(0) - yv) ** 2).mean(1))

def scalars(loc, scale, idx, seed=0):
    torch.manual_seed(seed); out = {}
    e = loc[:, idx['eta_e'][0]:idx['eta_e'][1]] + scale[:, idx['eta_e'][0]:idx['eta_e'][1]] * torch.randn(4000, loc.shape[0], 1)
    out['sigma_e'] = torch.sigmoid(e).mean(0).squeeze(-1).numpy()
    if 'eta_s' in idx:
        s = loc[:, idx['eta_s'][0]:idx['eta_s'][1]] + scale[:, idx['eta_s'][0]:idx['eta_s'][1]] * torch.randn(4000, loc.shape[0], 1)
        out['sigma_u'] = ((S_LO + (S_HI - S_LO) * torch.sigmoid(s)) ** 2).mean(0).squeeze(-1).numpy()
    return out

# ---------------------------------------------------------------- run
rows = []; PEROBS = {}
VARIANTS = os.environ.get('VARIANTS', 'a,c').split(',')   # a: BNN-RE; b: sigma_u-updating variant; c: tempered sigma_e prior
CUMU = os.environ.get('CUMU', '1') == '1'                    # CUMU=0: consecutive schedule only (long horizon)
def log(method, strategy, k_eval, loc, scale, idx, hid, use_re):
    ev = predict_eval(loc, scale, idx, hid, use_re, Xte[:, k_eval], yte[:, k_eval], Ste[:, k_eval], seed=1000 + k_eval, keep=PEROBS, key=f'{method}|{strategy}|{k_eval + 1}')
    sc = scalars(loc, scale, idx)
    for r in range(R):
        rows.append(dict(form=form, rep=r, method=method, strategy=strategy, eval_block=k_eval + 1,
                         width=float(ev['width'][r]), cover=float(ev['cover'][r]), mse=float(ev['mse'][r]),
                         sigma_e=float(sc['sigma_e'][r]), sigma_u=float(sc['sigma_u'][r]) if 'sigma_u' in sc else np.nan))
    return rows[-R:]

def cat_blocks(ks):
    return torch.cat([Xte[:, k] for k in ks], 1), torch.cat([yte[:, k] for k in ks], 1), torch.cat([Ste[:, k] for k in ks], 1)

for method, use_re in (('BNN', False), ('BNN-RE', True)):
    t1 = time.time()
    loc0, sc0, idx0 = fit(Xtr, ytr, Str, HID, use_re, None, False, STEPS, LR, seed=11)
    log(method, 'initial', 0, loc0, sc0, idx0, HID, use_re)
    print(f'[{form}] {method} initial fit {time.time() - t1:.0f}s', flush=True)
    variants = [(v, v == 'b', v == 'c') for v in VARIANTS] if use_re else [('-', False, False)]
    for var, reset, temper in variants:
        name = method if method == 'BNN' else f'BNN-RE({var})'
        prev = (loc0, sc0, idx0); n_prev = Xtr.shape[1]; upd1 = None
        for k in range(1, KB):
            pr = recycled(*prev, temper_n=n_prev if temper else None)
            loc, sc, idx = fit(Xte[:, k - 1], yte[:, k - 1], Ste[:, k - 1], HID, use_re, pr, reset, STEPS, LR, seed=20 + k)
            r_ = log(name, 'cons', k, loc, sc, idx, HID, use_re)
            if k == 1: upd1 = [dict(x, strategy='cumu') for x in r_]; PEROBS[f'{name}|cumu|2'] = PEROBS[f'{name}|cons|2']
            prev = (loc, sc, idx); n_prev = N0 + BLK
        rows.extend(upd1)
        for k in (range(2, KB) if CUMU else []):
            xc, yc, sc_ = cat_blocks(range(k))
            loc, sc, idx = fit(xc, yc, sc_, HID, use_re, recycled(loc0, sc0, idx0, temper_n=Xtr.shape[1] if temper else None), reset, STEPS, LR, seed=30 + k)
            log(name, 'cumu', k, loc, sc, idx, HID, use_re)
        print(f'[{form}] {name} updates done {time.time() - t1:.0f}s', flush=True)
df = pd.DataFrame(rows)
for r in range(R):
    df[df.rep == r].to_csv(f'{out}/sim_{form}{TAG}_rep{r}.csv', index=False)
np.savez_compressed(f'{out}/perobs_{form}{TAG}.npz', **PEROBS)
print(f'[{form}] finished {R} replicates, {KB} blocks, in {time.time() - t0:.0f}s', flush=True)
