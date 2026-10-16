"""
Corrected online BNN-RE library.

Fixes relative to the original notebooks:
 (1) Posterior recycling for constrained parameters (residual SD sigma_e, random-effect SD latent s)
     is done on the unconstrained scale with the matching bijection, so the recycled prior is exactly the
     mean-field marginal learned at the previous step.  The original code used AutoDiagonalNormal.get_posterior()
     moments (unconstrained/logit scale) as if they were on the constrained scale.
 (2) Random-effect priors in the "update u" strategy are aligned to sow sources by name.  The original code took
     the first k entries of the training random-effect vector regardless of which sources were present.
 (3) The random-effect SD prior is written as sigma_u = s^2 with s ~ Uniform(0.4, 0.8) (positive; the original
     used Uniform(-0.8, -0.4) and squared it, which gives the identical distribution for sigma_u).
"""
import math, numpy as np, torch, pyro, pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

S_LO, S_HI = 0.4, 0.8          # s ~ U(S_LO, S_HI), sigma_u = s^2  in (0.16, 0.64)
TAU_LOG = math.log(5.0)        # 5% nursery mortality action threshold on the log scale

def sigmoid(t): return torch.sigmoid(t)

class Spec:
    """Prior specification for one model fit."""
    def __init__(self, in_dim, hid, levels_seen, levels_new, use_re=True, net_prior=None, re_seen_prior=None,
                 sig_prior=None, s_prior=None, reset_re=False, prior_scale=1.0, scale_prior='bounded'):
        self.scale_prior = scale_prior        # 'bounded': s~U(0.4,0.8), sigma_u=s^2, sigma_e~Beta(3,3); 'halfnormal': sigma_u~HN(0.5), sigma_e~HN(1)
        self.in_dim, self.hid, self.use_re = in_dim, hid, use_re
        self.levels_seen, self.levels_new = list(levels_seen), list(levels_new)
        self.levels = self.levels_seen + self.levels_new
        self.net_prior = net_prior            # None -> N(0, prior_scale); else dict name->(loc, scale)
        self.re_seen_prior = re_seen_prior    # None -> hierarchical N(0, sigma_u); else (loc, scale) aligned to levels_seen
        self.sig_prior = sig_prior            # None -> Beta(3,3) on sigma_e; else (loc, scale) for eta_e, sigma_e = sigmoid(eta_e)
        self.s_prior = s_prior                # None -> U(S_LO,S_HI) on s; else (loc, scale) for eta_s, s = S_LO+(S_HI-S_LO)sigmoid(eta_s)
        self.reset_re = reset_re
        self.prior_scale = prior_scale

    def net_shapes(self):
        if self.hid > 0:
            return {'w1': (self.hid, self.in_dim), 'b1': (self.hid,), 'w2': (1, self.hid), 'b2': (1,)}
        return {'beta': (self.in_dim,), 'b0': (1,)}

def make_model(spec: Spec):
    shapes = spec.net_shapes()
    def model(x, z, y=None):
        net = {}
        for name, shp in shapes.items():
            if spec.net_prior is None:
                d = dist.Normal(torch.zeros(shp), spec.prior_scale * torch.ones(shp))
            else:
                loc, sc = spec.net_prior[name]; d = dist.Normal(loc, sc)
            net[name] = pyro.sample(name, d.to_event(len(shp)))
        if spec.hid > 0:
            h = torch.tanh(x @ net['w1'].T + net['b1'])
            mu = (h @ net['w2'].T + net['b2']).squeeze(-1)
        else:
            mu = x @ net['beta'] + net['b0']
        if spec.use_re:
            if spec.scale_prior == 'bounded':
                if spec.s_prior is None:
                    s = pyro.sample("s", dist.Uniform(S_LO, S_HI))
                else:
                    eta_s = pyro.sample("eta_s", dist.Normal(*spec.s_prior)); s = S_LO + (S_HI - S_LO) * sigmoid(eta_s)
                sig_u = s ** 2
            else:
                if spec.s_prior is None:
                    sig_u = pyro.sample("sig_u_hn", dist.HalfNormal(0.5))
                else:
                    eta_s = pyro.sample("eta_s", dist.Normal(*spec.s_prior)); sig_u = torch.exp(eta_s)
            parts = []
            n_seen, n_new = len(spec.levels_seen), len(spec.levels_new)
            if n_seen > 0:
                if spec.re_seen_prior is not None and not spec.reset_re:
                    loc, sc = spec.re_seen_prior
                    parts.append(pyro.sample("re_seen", dist.Normal(loc, sc).to_event(1)))
                else:
                    parts.append(pyro.sample("re_seen", dist.Normal(torch.zeros(n_seen), sig_u * torch.ones(n_seen)).to_event(1)))
            if n_new > 0:
                parts.append(pyro.sample("re_new", dist.Normal(torch.zeros(n_new), sig_u * torch.ones(n_new)).to_event(1)))
            re = torch.cat(parts, -1)
            mu = mu + z @ re
        if spec.scale_prior == 'bounded':
            if spec.sig_prior is None:
                sigma = pyro.sample("sigma", dist.Beta(3., 3.))
            else:
                eta_e = pyro.sample("eta_e", dist.Normal(*spec.sig_prior)); sigma = sigmoid(eta_e)
        else:
            if spec.sig_prior is None:
                sigma = pyro.sample("sigma_hn", dist.HalfNormal(1.0))
            else:
                eta_e = pyro.sample("eta_e", dist.Normal(*spec.sig_prior)); sigma = torch.exp(eta_e)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
    return model

JIT = False
def fit(spec, x, z, y, seed, lr=0.005, steps=2000):
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    model = make_model(spec); guide = AutoDiagonalNormal(model)
    elbo = JitTrace_ELBO(ignore_jit_warnings=True) if JIT else Trace_ELBO()
    svi = SVI(model, guide, pyro.optim.Adam({"lr": lr}), elbo)
    for _ in range(steps):
        loss = svi.step(x, z, y)
    return model, guide, loss / x.shape[0]

def unconstrained_posterior(guide):
    """site name -> (loc, scale) of the mean-field Gaussian on the UNCONSTRAINED scale."""
    out = {}
    locs = list(guide._unpack_latent(guide.loc.detach())); scs = list(guide._unpack_latent(guide.scale.detach()))
    for (site, l), (_, s) in zip(locs, scs):
        out[site['name']] = (l.reshape(site['value'].shape).clone(), s.reshape(site['value'].shape).clone())
    return out

def posterior_samples(guide, spec, S=5000, seed=0):
    """Draw S constrained posterior samples (dict of tensors with leading dim S)."""
    g = torch.Generator().manual_seed(seed)
    loc, scale = guide.loc.detach(), guide.scale.detach()
    eps = torch.randn((S, loc.numel()), generator=g)
    lat = loc + scale * eps
    out = {}
    for site, val in guide._unpack_latent(lat):   # val has shape (S, *site_shape) because lat has leading S
        name = site['name']; v = val.reshape((S,) + site['value'].shape)
        hn = (spec.scale_prior != 'bounded')
        if name == 'sigma':      out['sigma'] = sigmoid(v)                       # Beta support -> sigmoid bijection
        elif name == 's':        out['s'] = S_LO + (S_HI - S_LO) * sigmoid(v)    # interval bijection
        elif name == 'sigma_hn': out['sigma'] = torch.exp(v)                     # positive support -> exp bijection
        elif name == 'sig_u_hn': out['s'] = torch.sqrt(torch.exp(v))             # stored as s with sigma_u = s^2 for downstream code
        elif name == 'eta_e':    out['sigma'] = torch.exp(v) if hn else sigmoid(v)
        elif name == 'eta_s':    out['s'] = torch.sqrt(torch.exp(v)) if hn else S_LO + (S_HI - S_LO) * sigmoid(v)
        else: out[name] = v
    return out

def predict_draws(samples, spec, x, z):
    """Posterior predictive draws of y: (S, N)."""
    if spec.hid > 0:
        h = torch.tanh(torch.einsum('np,shp->snh', x, samples['w1']) + samples['b1'].unsqueeze(1))
        mu = torch.einsum('snh,soh->sno', h, samples['w2']).squeeze(-1) + samples['b2']
    else:
        mu = torch.einsum('np,sp->sn', x, samples['beta']) + samples['b0']
    if spec.use_re:
        parts = [samples[k] for k in ('re_seen', 're_new') if k in samples]
        re = torch.cat(parts, -1)
        mu = mu + torch.einsum('nk,sk->sn', z, re)
        unseen = (z.sum(1) == 0)                     # sources absent from the model: draw u from the hierarchical prior N(0, sigma_u)
        if unseen.any():
            g0 = torch.Generator().manual_seed(777)
            sig_u = (samples['s'] ** 2).reshape(-1, 1)
            mu = mu + unseen.float().unsqueeze(0) * sig_u * torch.randn(mu.shape, generator=g0)
    sigma = samples['sigma'].reshape(-1, 1)
    S = mu.shape[0]
    g = torch.Generator().manual_seed(12345)
    return mu + sigma * torch.randn(mu.shape, generator=g), mu

def score(draws, y, alpha=0.05):
    """Per-observation metrics from predictive draws (S,N) and truth y (N,)."""
    d = draws.numpy(); yv = y.numpy()
    lo = np.percentile(d, 100 * alpha / 2, axis=0); hi = np.percentile(d, 100 * (1 - alpha / 2), axis=0)
    mean = d.mean(0); width = hi - lo
    cover = ((yv > lo) & (yv < hi)).astype(float)
    iscore = width + (2 / alpha) * (lo - yv) * (yv < lo) + (2 / alpha) * (yv - hi) * (yv > hi)
    ds = np.sort(d, axis=0); S = d.shape[0]; i = np.arange(1, S + 1)[:, None]
    e_abs_diff = (2.0 / S ** 2) * ((2 * i - S - 1) * ds).sum(0)
    crps = np.abs(d - yv[None, :]).mean(0) - 0.5 * e_abs_diff
    p_exceed = (d > TAU_LOG).mean(0)
    return dict(lo=lo, hi=hi, mean=mean, width=width, cover=cover, iscore=iscore, crps=crps, p_exceed=p_exceed,
                sqerr=(mean - yv) ** 2, y=yv)

def zmat(sources, levels):
    idx = {s: j for j, s in enumerate(levels)}
    Z = torch.zeros(len(sources), len(levels))
    for i, s in enumerate(sources):
        if s in idx: Z[i, idx[s]] = 1.0
    return Z

def recycled_spec(prev_spec, prev_guide, new_sources, reset_re):
    """Build the Spec of an update model from the previous model (posterior -> prior)."""
    post = unconstrained_posterior(prev_guide)
    net_prior = {k: post[k] for k in prev_spec.net_shapes()}
    sig_prior = post['sigma'] if 'sigma' in post else (post['sigma_hn'] if 'sigma_hn' in post else post['eta_e'])
    if not prev_spec.use_re:
        return Spec(prev_spec.in_dim, prev_spec.hid, [], [], use_re=False, net_prior=net_prior, sig_prior=sig_prior, scale_prior=prev_spec.scale_prior)
    s_prior = post['s'] if 's' in post else (post['sig_u_hn'] if 'sig_u_hn' in post else post['eta_s'])
    seen = prev_spec.levels                                  # all sources known to the previous model
    new = [s for s in dict.fromkeys(new_sources) if s not in set(seen)]
    re_loc = torch.cat([post[k][0] for k in ('re_seen', 're_new') if k in post]); re_sc = torch.cat([post[k][1] for k in ('re_seen', 're_new') if k in post])
    return Spec(prev_spec.in_dim, prev_spec.hid, seen, new, use_re=True, net_prior=net_prior,
                re_seen_prior=(re_loc, re_sc), sig_prior=sig_prior, s_prior=s_prior, reset_re=reset_re, scale_prior=prev_spec.scale_prior)
