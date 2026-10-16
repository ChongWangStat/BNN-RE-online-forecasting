import time, torch, numpy as np, pyro, pyro.distributions as dist, torch.nn as nn
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from prep_data import *
torch.set_num_threads(1)
train, test, pred_cols = real_data_split()
sc = Scaler(torch.tensor(train[pred_cols].values.astype(np.float32)))
x, y = xy(train, pred_cols); x = sc(x)
levels = sorted(train.SowSource.unique()); z = zmat(train.SowSource.tolist(), levels)
print('train', x.shape, 'sources', len(levels))

class BNNRE(PyroModule):  # student's original model, verbatim logic
    def __init__(self, in_dim, hid_dim, n_cluster, prior_scale=1.):
        super().__init__(); self.cluster = n_cluster; self.act = nn.Tanh()
        self.weight1 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.bias1 = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.weight2 = PyroSample(dist.Normal(0., prior_scale).expand([1, hid_dim]).to_event(2))
        self.bias2 = PyroSample(dist.Normal(0., prior_scale).expand([1]).to_event(1))
    def forward(self, x, z, y=None):
        h = self.act(x @ self.weight1.T + self.bias1); mu = (h @ self.weight2.T + self.bias2).squeeze(-1)
        sigma_re = pyro.sample("sigma_re", dist.Uniform(-.8, -.4))
        re = pyro.sample("re", dist.Normal(0., sigma_re**2).expand([self.cluster]).to_event(1))
        mure = mu + z @ re
        sigma = pyro.sample("sigma", dist.Beta(3, 3))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mure, sigma), obs=y)
        return mure

pyro.clear_param_store(); pyro.set_rng_seed(567)
model = BNNRE(x.shape[1], 20, len(levels)); guide = AutoDiagonalNormal(model)
svi = SVI(model, guide, pyro.optim.Adam({"lr": 0.005}), Trace_ELBO())
t0 = time.time()
for ep in range(2000):
    loss = svi.step(x=x, z=z, y=y)
print(f'2000 SVI steps: {time.time()-t0:.1f}s; final loss/N {loss/x.shape[0]:.3f}')
post = guide.get_posterior(); m = post.mean.detach(); s = post.stddev.detach()
med = guide.median(); q = guide.quantiles([0.025, 0.5, 0.975])
hid, ind = 20, x.shape[1]; i_sre = hid*ind + 2*hid + 1
print(f"sigma   : recycled get_posterior().mean = {m[-1]:.3f} (sd {s[-1]:.3f})  | constrained median = {med['sigma'].item():.3f}  | logit(median) = {torch.logit(med['sigma']).item():.3f}")
print(f"sigma_re: recycled get_posterior().mean = {m[i_sre]:.3f} (sd {s[i_sre]:.3f}) | constrained median s = {med['sigma_re'].item():.3f} -> sigma_u = s^2 = {med['sigma_re'].item()**2:.3f}")
print(f"          recycled value used as s in update: s^2 = {m[i_sre].item()**2:.3f}  (vs true learned sigma_u {med['sigma_re'].item()**2:.3f})")
print(f"          95% CrI of sigma_u: [{q['sigma_re'][0].item()**2:.3f}, {q['sigma_re'][2].item()**2:.3f}]  ; sigma_e CrI [{q['sigma'][0].item():.3f},{q['sigma'][2].item():.3f}]")
re_m = med['re'].numpy(); print('random effects: sd of posterior medians', re_m.std().round(3), 'range', re_m.min().round(2), re_m.max().round(2))
t0 = time.time(); pr = Predictive(model, guide=guide, num_samples=5000)
xt, yt = xy(test[test.week_num <= 5], pred_cols); xt = sc(xt); zt = zmat(test[test.week_num <= 5].SowSource.tolist(), levels)
pp = pr(x=xt, z=zt)['obs'].numpy(); print(f'predictive 5000 draws: {time.time()-t0:.1f}s')
lo, hi = np.percentile(pp, [2.5, 97.5], axis=0); print('test1 width', (hi-lo).mean().round(3), 'cvrg', ((yt.numpy() > lo) & (yt.numpy() < hi)).mean().round(3))
np.save('re_medians_seed567.npy', re_m)
