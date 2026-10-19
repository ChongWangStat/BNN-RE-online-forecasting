"""Online BNN / BNN-RE prediction intervals with posterior recycling (corrected implementation).

Model, priors, variational family, optimiser and all hyperparameters follow the original
notebooks (project2/real_data_clean_code). Four implementation points differ, all of them
needed to make the code do what the manuscript describes:
  1. Recycled priors for the constrained parameters (sigma_e, and s where sigma_u = s^2) are
     placed on the unconstrained scale of the mean-field guide and mapped back with the same
     bijection Pyro uses (logit for (0,1); scaled logit for (0.4, 0.8)).
  2. Source effects are recycled by source identity: the random-effect vector always covers the
     full list of training sources, and design matrices are built by name against that list.
  3. Consecutive updates recycle the posterior of the preceding update; cumulative updates
     recycle the initial posterior and use all blocks observed so far.
  4. (simulation, see run_sim.py) test subjects reuse their own training effect direction.
Note: Pyro keeps AutoDiagonalNormal parameters in a global store under a fixed name, so a guide must be
used for prediction immediately after it is fitted; recycled priors are read from the guide object
(unconstrained_posterior), which returns that guide's own parameters.
"""
import numpy as np, torch, pyro, pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

S_LO, S_HI = 0.4, 0.8          # s ~ Uniform(S_LO, S_HI), sigma_u = s^2   (original: Uniform(-.8,-.4)^2)
W_SCALE = 1.0                  # p(theta) = N(0, 1)                        (original: prior_scale=1)

def make_model(in_dim, hid, n_src, use_re, prior=None, reset_re=False):
    """prior=None -> initial model.  prior=dict of (loc, scale) on the unconstrained scale -> update model.
    reset_re=True (variant b): recycle the s latent and redraw the source effects from N(0, s^2).
    reset_re=False (variant a): recycle each source effect's own marginal; sigma_u fixed (not sampled)."""
    def model(x, z, y=None):
        if hid == 0:   # linear mean function (Bayesian linear mixed model when use_re): mu = x w + b
            if prior is None:
                w2 = pyro.sample('w2', dist.Normal(0., W_SCALE).expand([1, in_dim]).to_event(2))
                b2 = pyro.sample('b2', dist.Normal(0., W_SCALE).expand([1]).to_event(1))
            else:
                w2 = pyro.sample('w2', dist.Normal(*prior['w2']).to_event(2))
                b2 = pyro.sample('b2', dist.Normal(*prior['b2']).to_event(1))
            mu = (x @ w2.transpose(-1, -2) + b2.unsqueeze(-2)).squeeze(-1)
        else:
            if prior is None:
                w1 = pyro.sample('w1', dist.Normal(0., W_SCALE).expand([hid, in_dim]).to_event(2))
                b1 = pyro.sample('b1', dist.Normal(0., W_SCALE).expand([hid]).to_event(1))
                w2 = pyro.sample('w2', dist.Normal(0., W_SCALE).expand([1, hid]).to_event(2))
                b2 = pyro.sample('b2', dist.Normal(0., W_SCALE).expand([1]).to_event(1))
            else:
                w1 = pyro.sample('w1', dist.Normal(*prior['w1']).to_event(2))
                b1 = pyro.sample('b1', dist.Normal(*prior['b1']).to_event(1))
                w2 = pyro.sample('w2', dist.Normal(*prior['w2']).to_event(2))
                b2 = pyro.sample('b2', dist.Normal(*prior['b2']).to_event(1))
            # batch-safe algebra so that Predictive(parallel=True) can draw all samples at once
            h = torch.tanh(x @ w1.transpose(-1, -2) + b1.unsqueeze(-2))
            mu = (h @ w2.transpose(-1, -2) + b2.unsqueeze(-2)).squeeze(-1)
        if use_re:
            if prior is None:
                s = pyro.sample('s', dist.Uniform(S_LO, S_HI))
                re = pyro.sample('re', dist.Normal(torch.zeros(n_src), (s ** 2) * torch.ones(n_src)).to_event(1))
            elif reset_re:
                eta_s = pyro.sample('eta_s', dist.Normal(*prior['eta_s']))
                s = S_LO + (S_HI - S_LO) * torch.sigmoid(eta_s)
                re = pyro.sample('re', dist.Normal(torch.zeros(n_src), (s ** 2) * torch.ones(n_src)).to_event(1))
            else:
                re = pyro.sample('re', dist.Normal(*prior['re']).to_event(1))
            mu = mu + torch.matmul(z, re.unsqueeze(-1)).squeeze(-1)
        if prior is None:
            sigma = pyro.sample('sigma', dist.Beta(3., 3.))
        else:
            eta_e = pyro.sample('eta_e', dist.Normal(*prior['eta_e']))
            sigma = torch.sigmoid(eta_e)
        with pyro.plate('data', x.shape[0]):
            pyro.sample('obs', dist.Normal(mu, sigma), obs=y)
        return mu
    return model

from pyro.infer import JitTrace_ELBO
def fit(model, x, z, y, seed, lr=0.005, steps=2000, jit=False, init=None):
    """init=None: Pyro's default initialisation (loc at the prior median, scale 0.1).  init=dict of (loc, scale) per
    site on the unconstrained scale (the recycled prior): warm start, the variational parameters begin at the previous
    posterior and the optimisation continues from there."""
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    guide = AutoDiagonalNormal(model)
    if init is not None:
        guide(x, z, y)                                   # create the parameters
        with torch.no_grad():
            loc = guide.loc.detach().clone(); scale = guide.scale.detach().clone()
            for (site, lv), (_, sv) in zip(guide._unpack_latent(loc), guide._unpack_latent(scale)):
                name = {'sigma': 'eta_e', 's': 'eta_s'}.get(site['name'], site['name'])
                if name in init:
                    lv.copy_(init[name][0].reshape(lv.shape)); sv.copy_(init[name][1].reshape(sv.shape))
            guide.loc.data.copy_(loc); guide.scale_unconstrained.data.copy_(torch.log(scale))
    elbo = JitTrace_ELBO(ignore_jit_warnings=True) if jit else Trace_ELBO()
    svi = SVI(model, guide, pyro.optim.Adam({'lr': lr}), loss=elbo)
    loss = None
    for _ in range(steps):
        loss = svi.step(x, z, y) / x.shape[0]
    return guide, float(loss)

def unconstrained_posterior(guide):
    """Per-site (loc, scale) of the mean-field guide on the unconstrained scale.
    'sigma' (Beta site) -> 'eta_e' (logit scale); 's' (Uniform site) -> 'eta_s' (scaled-logit scale)."""
    post = guide.get_posterior()
    loc, scale = post.mean.detach(), post.stddev.detach()
    out = {}
    for (site, lv), (_, sv) in zip(guide._unpack_latent(loc), guide._unpack_latent(scale)):
        name = {'sigma': 'eta_e', 's': 'eta_s'}.get(site['name'], site['name'])
        out[name] = (lv.clone(), sv.clone())
    return out

def recycled_prior(guide):
    return unconstrained_posterior(guide)

def predict(model, guide, x, z, seed, n_draws=5000):
    pyro.set_rng_seed(seed)
    draws = Predictive(model, guide=guide, num_samples=n_draws, return_sites=('obs',), parallel=True)(x, z)['obs'].detach()
    return draws.reshape(n_draws, -1).numpy()  # [n_draws, n]

def evaluate(draws, y, alpha=0.05):
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    mean = draws.mean(axis=0); yv = y.numpy()
    return dict(width=float(np.mean(hi - lo)), cover=float(np.mean((lo < yv) & (yv < hi))),
                mse=float(np.mean((mean - yv) ** 2)), lo=lo, hi=hi, mean=mean, y=yv)

def posterior_scalars(guide, prior_is_initial):
    """Posterior mean of sigma_e (and sigma_u when present) from the guide, for monitoring."""
    pyro.set_rng_seed(0)
    post = unconstrained_posterior(guide)
    out = {}
    if 'eta_e' in post:
        d = dist.Normal(*post['eta_e']).sample((4000,)); out['sigma_e'] = float(torch.sigmoid(d).mean())
    if 'eta_s' in post:
        d = dist.Normal(*post['eta_s']).sample((4000,)); out['sigma_u'] = float(((S_LO + (S_HI - S_LO) * torch.sigmoid(d)) ** 2).mean())
    return out
