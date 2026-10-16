"""Reliability diagram of exceedance probabilities (Fig. S2), per-cell SE table (Table S3), paired differences, prior sensitivity (Table S4)."""
import glob, numpy as np, pandas as pd, matplotlib
# FRONTIERS-FONTS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})  # >= 8 pt at 180 mm final width
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bnnre_lib import TAU_LOG
import os
P = os.environ.get('BNNRE_OUT', 'outputs'); os.makedirs(f'{P}/figures', exist_ok=True); os.makedirs(f'{P}/tables', exist_ok=True)
N = {}
# ---- reliability diagram ----
npz = {f: np.load(f, allow_pickle=True) for f in sorted(glob.glob('results_real/perobs_seed*.npz'))}
def get(method, step, strat, key): return np.stack([z[f'{method}|{step}|{strat}|{key}'] for z in npz.values()])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
sel = [('BNN-RE', 'cumu_a', 'BNN-RE, cumu., carry u', '#E45756', '-'), ('BNN-RE', 'cumu_b', 'BNN-RE, cumu., re-learn', '#B2182B', '--'), ('BNN', 'cumu_-', 'BNN, cumulative', '#7f7f7f', '-'), ('LMM-RE', 'cumu_a', 'LMM-RE, cumu., carry u', '#4C78A8', '-')]
for method, strat, lab, c, ls in sel:
    ps, ys = [], []
    for k in range(5):
        st = 'initial' if k == 0 else strat
        ps.append(get(method, k, st, 'p_exceed').mean(0)); ys.append((get(method, k, st, 'y')[0] > TAU_LOG).astype(float))
    p = np.concatenate(ps); y = np.concatenate(ys)
    edges = np.linspace(0, 1, 6); idx = np.clip(np.digitize(p, edges) - 1, 0, 4)
    xs = [p[idx == b].mean() for b in range(5) if (idx == b).sum() >= 5]; obs = [y[idx == b].mean() for b in range(5) if (idx == b).sum() >= 5]; ns = [(idx == b).sum() for b in range(5) if (idx == b).sum() >= 5]
    ece = sum(n * abs(o - x) for n, o, x in zip(ns, obs, xs)) / len(p)
    axes[0].plot(xs, obs, marker='o', color=c, ls=ls, label=f'{lab} (ECE {ece:.03f})', ms=4)
    tag = {'cumu_a': 'CumuA', 'cumu_b': 'CumuB', 'cumu_-': 'BNN'}[strat] if method != 'LMM-RE' else 'LMM'
    N[f'ece{tag}'] = f'{ece:.3f}'
    if method == 'BNN-RE' and strat == 'cumu_a':
        axes[1].hist(p, bins=20, color=c, alpha=0.7); axes[1].set_xlabel('Predicted probability of exceeding 5% mortality'); axes[1].set_ylabel('Number of groups (2023, pooled)')
axes[0].plot([0, 1], [0, 1], 'k:', lw=0.8); axes[0].set_xlabel('Mean predicted probability (quintile bins)'); axes[0].set_ylabel('Observed proportion above 5%'); axes[0].legend(loc='upper left', fontsize=10, frameon=False)
fig.tight_layout(); fig.savefig(f'{P}/figures/fig_reliability.pdf'); fig.savefig(f'{P}/figures/fig_reliability.png', dpi=300); print('reliability done', N)
# ---- per-cell SE table (same layout as Table 1) ----
files = sorted(glob.glob('results_real/summary_seed*.csv')); df = pd.concat([pd.read_csv(f) for f in files]); R = df.seed.nunique()
g = df.groupby(['method', 'strategy', 'step']).agg(width_se=('width', lambda v: v.std(ddof=1) / np.sqrt(len(v))), cover_se=('cover', lambda v: v.std(ddof=1) / np.sqrt(len(v)))).reset_index()
def val(m, st, k, col):
    r = g[(g.method == m) & (g.strategy == st) & (g.step == k)]; return f"{r[col].values[0]:.3f}" if len(r) else '--'
cols = [('BNN', lambda sch: sch + '_-'), ('LMM-RE', lambda sch: sch + '_a'), ('LMM-RE', lambda sch: sch + '_b'), ('BNN-RE', lambda sch: sch + '_a'), ('BNN-RE', lambda sch: sch + '_b'), ('BNN-RE-batch', lambda sch: 'batch')]
def cells(k, sch, col):
    out = []
    for m, f in cols:
        if k == 0: out.append(val('BNN-RE' if m == 'BNN-RE-batch' else m, 'initial', 0, col))
        else: out.append(val(m, f(sch), k, col))
    return out
hdr = r' & & \textbf{BNN} & \textbf{LMM-RE}$^{(a)}$ & \textbf{LMM-RE}$^{(b)}$ & \textbf{BNN-RE}$^{(a)}$ & \textbf{BNN-RE}$^{(b)}$ & \textbf{BNN-RE}$^{\dagger}$ \\'
L = [r'\begin{tabular}{cc|cccccc}', r'\hline']
for head, col in [('Monte Carlo SE of average interval width', 'width_se'), ('Monte Carlo SE of average coverage rate', 'cover_se')]:
    L += [r' & & \multicolumn{6}{c}{\textbf{' + head + r'}} \\', r'\hline', hdr, r'\hline']
    L.append(r'\multicolumn{2}{c|}{\textbf{Initial}} & ' + ' & '.join(cells(0, None, col)) + r' \\')
    L.append(r'\multicolumn{2}{c|}{\textbf{Update \#1}} & ' + ' & '.join(cells(1, 'cons', col)) + r' \\')
    for k in range(2, 5):
        L.append(r'\multirow{2}{*}{\textbf{Update \#' + str(k) + r'}} & Cons. & ' + ' & '.join(cells(k, 'cons', col)) + r' \\')
        L.append(r' & Cumu. & ' + ' & '.join(cells(k, 'cumu', col)) + r' \\')
    L.append(r'\hline')
L.append(r'\end{tabular}'); open(f'{P}/tables/table_real_main_se.tex', 'w').write('\n'.join(L))
# ---- paired differences across seeds (averaged over updates 1-4) ----
def per_seed(m, s): return df[(df.method == m) & (df.strategy == s) & (df.step >= 1)].groupby('seed')[['width', 'iscore', 'crps']].mean()
a = per_seed('BNN-RE', 'cumu_a'); b = per_seed('BNN', 'cumu_-'); l = per_seed('LMM-RE', 'cumu_a'); bt = per_seed('BNN-RE-batch', 'batch')
for name, d in [('BNNREminusBNN', a - b), ('LMMminusBNNRE', l - a), ('BatchminusBNNRE', bt - a)]:
    for col in ['width', 'iscore', 'crps']:
        N[f'pd{name}{col.capitalize()}'] = f"{d[col].mean():.3f}"; N[f'pd{name}{col.capitalize()}SE'] = f"{d[col].std(ddof=1) / np.sqrt(len(d)):.3f}"
# ---- prior sensitivity ----
pf = sorted(glob.glob('results_prior/prior_sens_seed*.csv'))
if pf:
    ps = pd.concat([pd.read_csv(f) for f in pf]); Rp = ps.seed.nunique()
    agg = ps.groupby(['prior', 'step']).agg(width=('width', 'mean'), cover=('cover', 'mean'), iscore=('iscore', 'mean'), sigma_e=('sigma_e', 'mean'), sigma_u=('sigma_u', 'mean')).reset_index()
    L = [r'\begin{tabular}{l l ccccc}', r'\hline', r'Scale priors & Step & Width & Coverage & Interval score & $\hat\sigma_e$ & $\hat\sigma_u$ \\', r'\hline']
    for prior, lab in [('bounded', r'$s\sim$U(0.4,0.8), $\sigma_u=s^2$; $\sigma_e\sim$Beta(3,3)'), ('halfnormal', r'$\sigma_u\sim$HN(0.5); $\sigma_e\sim$HN(1)')]:
        for k in range(5):
            r = agg[(agg.prior == prior) & (agg.step == k)].iloc[0]
            L.append(f"{lab if k == 0 else ''} & {'Initial' if k == 0 else f'Update \\#{k} (Cumu., carry)'} & {r.width:.2f} & {r.cover:.3f} & {r.iscore:.2f} & {r.sigma_e:.2f} & {r.sigma_u:.2f} \\\\")
        L.append(r'\hline')
    L.append(r'\end{tabular}'); open(f'{P}/tables/table_prior_sens.tex', 'w').write('\n'.join(L))
    N['priorReps'] = str(Rp)
    for prior, tag in [('bounded', 'Bnd'), ('halfnormal', 'HN')]:
        sub = agg[(agg.prior == prior) & (agg.step >= 1)]; i0 = agg[(agg.prior == prior) & (agg.step == 0)].iloc[0]
        N[f'psW{tag}'] = f"{sub.width.mean():.2f}"; N[f'psC{tag}'] = f"{sub.cover.mean():.3f}"; N[f'psIS{tag}'] = f"{sub.iscore.mean():.2f}"; N[f'psSigE{tag}'] = f"{i0.sigma_e:.2f}"; N[f'psSigU{tag}'] = f"{i0.sigma_u:.2f}"; N[f'psWinit{tag}'] = f"{i0.width:.2f}"
    print(agg.round(3).to_string())
with open(f'{P}/numbers_extra.tex', 'w') as f:
    for k, v in N.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('macros', len(N))
