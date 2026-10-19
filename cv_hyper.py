"""Five-fold cross-validation of the BNN-RE hyperparameters on the training records.
Grid: learning rate {0.05, 0.01, 0.005, 0.001} x hidden units {5, 10, 20, 50} x epochs {1500, 2000, 2500}.
The three epoch values are evaluated as checkpoints of one 2,500-epoch run per (lr, hidden, fold).
Criterion: cross-validated mean squared error of the posterior predictive mean (log mortality); the interval
width and coverage of the 95% predictive interval are recorded as well.
Usage: python cv_hyper.py data.xlsx cv_results.csv [SEED]"""
import sys, time, numpy as np, pandas as pd, torch, pyro
from pyro.infer import SVI, JitTrace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from prep_real import load, Design
from bnnre_lib import make_model, predict, evaluate

xlsx, dest = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.set_num_threads(1)
LRS, HIDS, EPOCHS = [0.05, 0.01, 0.005, 0.001], [5, 10, 20, 50], [1500, 2000, 2500]
train, blocks, info = load(xlsx); D = Design(train)
x, y, z, _ = D.xyz(train); n, in_dim, n_src = x.shape[0], x.shape[1], len(D.levels)
rng = np.random.default_rng(seed); perm = rng.permutation(n); folds = np.array_split(perm, 5)
rows = []; t0 = time.time()
for hid in HIDS:
    for lr in LRS:
        for f, te_idx in enumerate(folds):
            tr_idx = np.setdiff1d(perm, te_idx)
            xt, yt, zt = x[tr_idx], y[tr_idx], z[tr_idx]; xe, ye, ze = x[te_idx], y[te_idx], z[te_idx]
            pyro.clear_param_store(); pyro.set_rng_seed(seed)
            model = make_model(in_dim, hid, n_src, True); guide = AutoDiagonalNormal(model)
            svi = SVI(model, guide, pyro.optim.Adam({'lr': lr}), loss=JitTrace_ELBO(ignore_jit_warnings=True))
            done = 0
            for ep in EPOCHS:
                for _ in range(ep - done): svi.step(xt, zt, yt)
                done = ep
                ev = evaluate(predict(model, guide, xe, ze, seed, 2000), ye)
                rows.append(dict(hidden=hid, lr=lr, epochs=ep, fold=f, mse=ev['mse'], width=ev['width'], cover=ev['cover']))
            print(f'hid {hid} lr {lr} fold {f} done {time.time() - t0:.0f}s  mse@2000={rows[-2]["mse"]:.4f}', flush=True)
        pd.DataFrame(rows).to_csv(dest, index=False)
df = pd.DataFrame(rows); df.to_csv(dest, index=False)
summ = df.groupby(['hidden', 'lr', 'epochs']).agg(mse=('mse', 'mean'), width=('width', 'mean'), cover=('cover', 'mean')).reset_index().sort_values('mse')
print(summ.head(12).to_string(index=False)); print('best:', summ.iloc[0].to_dict())
