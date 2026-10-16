"""Aggregate corrected real-data results -> LaTeX tables, numbers.tex macros, figures."""
import glob, numpy as np, pandas as pd, matplotlib
# FRONTIERS-FONTS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})  # >= 8 pt at 180 mm final width
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bnnre_lib import TAU_LOG
import os
P = os.environ.get('BNNRE_OUT', 'outputs'); os.makedirs(f'{P}/figures', exist_ok=True); os.makedirs(f'{P}/tables', exist_ok=True)
files = sorted(glob.glob('results_real/summary_seed*.csv')); df = pd.concat([pd.read_csv(f) for f in files]); R = df.seed.nunique()
print('replicates', R)
g = df.groupby(['method', 'strategy', 'step'])
agg = g.agg(width=('width', 'mean'), width_se=('width', lambda v: v.std(ddof=1) / np.sqrt(len(v))), cover=('cover', 'mean'),
            cover_se=('cover', lambda v: v.std(ddof=1) / np.sqrt(len(v))), iscore=('iscore', 'mean'), iscore_se=('iscore', lambda v: v.std(ddof=1) / np.sqrt(len(v))),
            crps=('crps', 'mean'), mse=('mse', 'mean'), sigma_e=('sigma_e', 'mean'), sigma_u=('sigma_u', 'mean')).reset_index()
agg.to_csv(f'{P}/tables/real_summary.csv', index=False)
pd.set_option('display.width', 250); print(agg.round(3).to_string())

def cell(m, s, k, col, se=True, nd=2):
    r = agg[(agg.method == m) & (agg.strategy == s) & (agg.step == k)]
    if len(r) == 0: return '--'
    v = r[col].values[0]; e = r[col + '_se'].values[0] if se and col + '_se' in r else None
    return f'{v:.{nd}f}' + (f' ({e:.{nd}f})' if e is not None else '')

# ---------------- Tables 1 and 2 in the original layout (rows = update step x scheme; columns = methods) -----------------
def val(m, st, k, col, nd):
    r = agg[(agg.method == m) & (agg.strategy == st) & (agg.step == k)]
    return f"{r[col].values[0]:.{nd}f}" if len(r) else '--'
def table_orig(col1, col2, nd1, nd2, head1, head2, fname):
    cols = [('BNN', lambda sch: 'initial' if sch is None else sch + '_-'), ('LMM-RE', lambda sch: 'initial' if sch is None else sch + '_a'), ('LMM-RE', lambda sch: 'initial' if sch is None else sch + '_b'),
            ('BNN-RE', lambda sch: 'initial' if sch is None else sch + '_a'), ('BNN-RE', lambda sch: 'initial' if sch is None else sch + '_b'), ('BNN-RE-batch', lambda sch: 'batch')]
    def cells(k, sch, col, nd):
        out = []
        for m, f in cols:
            if k == 0: out.append(val('BNN-RE' if m == 'BNN-RE-batch' else m, 'initial', 0, col, nd))
            else: out.append(val(m, f(sch), k, col, nd))
        return out
    hdr = r' & & \textbf{BNN} & \textbf{LMM-RE}$^{(a)}$ & \textbf{LMM-RE}$^{(b)}$ & \textbf{BNN-RE}$^{(a)}$ & \textbf{BNN-RE}$^{(b)}$ & \textbf{BNN-RE}$^{\dagger}$ \\'
    L = [r'\begin{tabular}{cc|cccccc}', r'\hline']
    for head, col, nd in [(head1, col1, nd1), (head2, col2, nd2)]:
        L.append(r' & & \multicolumn{6}{c}{\textbf{' + head + r'}} \\'); L.append(r'\hline'); L.append(hdr); L.append(r'\hline')
        L.append(r'\multicolumn{2}{c|}{\textbf{Initial}} & ' + ' & '.join(cells(0, None, col, nd)) + r' \\')
        L.append(r'\multicolumn{2}{c|}{\textbf{Update \#1}} & ' + ' & '.join(cells(1, 'cons', col, nd)) + r' \\')
        for k in range(2, 5):
            L.append(r'\multirow{2}{*}{\textbf{Update \#' + str(k) + r'}} & Cons. & ' + ' & '.join(cells(k, 'cons', col, nd)) + r' \\')
            L.append(r' & Cumu. & ' + ' & '.join(cells(k, 'cumu', col, nd)) + r' \\')
        L.append(r'\hline')
    L.append(r'\end{tabular}')
    open(f'{P}/tables/{fname}', 'w').write('\n'.join(L))
table_orig('width', 'cover', 2, 3, 'Average Interval Width', 'Average Interval Coverage Rate', 'table_real_main.tex')
table_orig('iscore', 'crps', 2, 3, 'Average Interval Score', 'Average CRPS', 'table_real_scores.tex')
semax = agg[agg.step >= 0]; N0 = dict(seWidthMax=f"{semax.width_se.max():.3f}", seCoverMax=f"{semax.cover_se.max():.3f}", seISMax=f"{semax.iscore_se.max():.3f}")
# ---------------- per-observation: exceedance probabilities, forecast figure -----------------
npz = {f: np.load(f, allow_pickle=True) for f in sorted(glob.glob('results_real/perobs_seed*.npz'))}
def get(method, step, strat, key):
    return np.stack([z[f'{method}|{step}|{strat}|{key}'] for z in npz.values()])
from itertools import product
def auc(p, yb):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = yb.sum(); n0 = len(yb) - n1
    return (r[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 > 0 and n0 > 0 else np.nan
ex_rows = []
for method, strat in [('BNN', 'cons_-'), ('BNN', 'cumu_-'), ('LMM-RE', 'cons_a'), ('LMM-RE', 'cumu_a'), ('LMM-RE', 'cumu_b'), ('BNN-RE', 'cons_a'), ('BNN-RE', 'cons_b'), ('BNN-RE', 'cumu_a'), ('BNN-RE', 'cumu_b'), ('BNN-RE-batch', 'batch')]:
    P_all, Y_all = [], []
    for k in range(0, 5):
        st = 'initial' if k == 0 else strat
        if method == 'BNN-RE-batch' and k == 0: st = 'initial'; m = 'BNN-RE'
        else: m = method
        try:
            p = get(m, k, st, 'p_exceed').mean(0); y = get(m, k, st, 'y')[0]
        except KeyError: continue
        yb = (y > TAU_LOG).astype(int); flag = (p > 0.5).astype(int)
        sens = (flag[yb == 1] == 1).mean() if yb.sum() else np.nan; spec = (flag[yb == 0] == 0).mean() if (1 - yb).sum() else np.nan
        ex_rows.append(dict(method=method, strategy=strat, step=k, n=len(y), prevalence=yb.mean(), sens=sens, spec=spec, auc=auc(p, yb), brier=((p - yb) ** 2).mean()))
        P_all.append(p); Y_all.append(yb)
    p = np.concatenate(P_all); yb = np.concatenate(Y_all); flag = (p > 0.5).astype(int)
    ex_rows.append(dict(method=method, strategy=strat, step=99, n=len(yb), prevalence=yb.mean(), sens=(flag[yb == 1] == 1).mean(), spec=(flag[yb == 0] == 0).mean(), auc=auc(p, yb), brier=((p - yb) ** 2).mean()))
ex = pd.DataFrame(ex_rows); ex.to_csv(f'{P}/tables/exceedance.csv', index=False); print(ex.round(3).to_string())
# LaTeX table of pooled (all five test sets) exceedance performance + per-step AUC
L = [r'\begin{tabular}{ll ccccc ccccc}', r'\hline', r'Method & Strategy & \multicolumn{5}{c}{AUC by test set (initial, updates 1--4)} & \multicolumn{5}{c}{Pooled over the 2023 test sets} \\',
     r'\cline{3-7}\cline{8-12}', r' & & 1 & 2 & 3 & 4 & 5 & $n$ & Sens. & Spec. & AUC & Brier \\', r'\hline']
lab = {'cons_-': 'Consecutive', 'cumu_-': 'Cumulative', 'cons_a': 'Cons., carry $\\bm u$', 'cons_b': 'Cons., re-learn', 'cumu_a': 'Cumu., carry $\\bm u$', 'cumu_b': 'Cumu., re-learn', 'batch': 'Batch retrain'}
for (method, strat), sub in ex.groupby(['method', 'strategy'], sort=False):
    per = [f"{sub[sub.step == k].auc.values[0]:.2f}" if (sub.step == k).any() else '--' for k in range(5)]
    pool = sub[sub.step == 99].iloc[0]
    L.append(f"{method} & {lab[strat]} & " + ' & '.join(per) + f" & {int(pool.n)} & {pool.sens:.2f} & {pool.spec:.2f} & {pool.auc:.2f} & {pool.brier:.3f} \\\\")
L += [r'\hline', r'\end{tabular}']
open(f'{P}/tables/table_exceedance.tex', 'w').write('\n'.join(L))

# ---------------- numbers.tex macros -----------------
def num(m, s, k, col, nd=2): return f"{agg[(agg.method == m) & (agg.strategy == s) & (agg.step == k)][col].values[0]:.{nd}f}"
N = {'nReps': str(R)}; N.update(N0)
for m, tag in [('BNN', 'BNN'), ('BNN-RE', 'BNNRE'), ('LMM-RE', 'LMM')]:
    N[f'wInit{tag}'] = num(m, 'initial', 0, 'width'); N[f'cInit{tag}'] = num(m, 'initial', 0, 'cover', 3); N[f'isInit{tag}'] = num(m, 'initial', 0, 'iscore')
for s, tag in [('cons_a', 'ConsA'), ('cons_b', 'ConsB'), ('cumu_a', 'CumuA'), ('cumu_b', 'CumuB')]:
    for k in range(1, 5):
        N[f'w{tag}{"ABCD"[k-1]}'] = num('BNN-RE', s, k, 'width'); N[f'c{tag}{"ABCD"[k-1]}'] = num('BNN-RE', s, k, 'cover', 3); N[f'is{tag}{"ABCD"[k-1]}'] = num('BNN-RE', s, k, 'iscore')
        N[f'wLMM{tag}{"ABCD"[k-1]}'] = num('LMM-RE', s, k, 'width'); N[f'cLMM{tag}{"ABCD"[k-1]}'] = num('LMM-RE', s, k, 'cover', 3)
for s, tag in [('cons_-', 'Cons'), ('cumu_-', 'Cumu')]:
    for k in range(1, 5):
        N[f'wBNN{tag}{"ABCD"[k-1]}'] = num('BNN', s, k, 'width'); N[f'cBNN{tag}{"ABCD"[k-1]}'] = num('BNN', s, k, 'cover', 3)
for k in range(1, 5):
    N[f'wBatch{"ABCD"[k-1]}'] = num('BNN-RE-batch', 'batch', k, 'width'); N[f'cBatch{"ABCD"[k-1]}'] = num('BNN-RE-batch', 'batch', k, 'cover', 3); N[f'isBatch{"ABCD"[k-1]}'] = num('BNN-RE-batch', 'batch', k, 'iscore')
# averages over the four updates
for m, s, tag in [('BNN-RE', 'cons_a', 'ConsA'), ('BNN-RE', 'cons_b', 'ConsB'), ('BNN-RE', 'cumu_a', 'CumuA'), ('BNN-RE', 'cumu_b', 'CumuB'), ('BNN', 'cons_-', 'BNNCons'), ('BNN', 'cumu_-', 'BNNCumu'), ('LMM-RE', 'cumu_a', 'LMMCumuA'), ('LMM-RE', 'cons_a', 'LMMConsA'), ('BNN-RE-batch', 'batch', 'Batch')]:
    sub = agg[(agg.method == m) & (agg.strategy == s) & (agg.step >= 1)]
    N[f'avgW{tag}'] = f"{sub.width.mean():.2f}"; N[f'avgC{tag}'] = f"{sub.cover.mean():.3f}"; N[f'avgIS{tag}'] = f"{sub.iscore.mean():.2f}"; N[f'avgCRPS{tag}'] = f"{sub.crps.mean():.3f}"
init = df[(df.method == 'BNN-RE') & (df.step == 0)]
N['sigmaE'] = f"{init.sigma_e.mean():.2f}"; N['sigmaU'] = f"{init.sigma_u.mean():.2f}"; N['sigmaUsd'] = f"{init.sigma_u.std():.3f}"
N['sigmaELMM'] = f"{df[(df.method == 'LMM-RE') & (df.step == 0)].sigma_e.mean():.2f}"; N['sigmaULMM'] = f"{df[(df.method == 'LMM-RE') & (df.step == 0)].sigma_u.mean():.2f}"
N['sigmaEBNN'] = f"{df[(df.method == 'BNN') & (df.step == 0)].sigma_e.mean():.2f}"
pooled = ex[(ex.step == 99)].set_index(['method', 'strategy'])
for (m, s), tag in [(('BNN-RE', 'cumu_a'), 'CumuA'), (('BNN-RE', 'cons_a'), 'ConsA'), (('BNN-RE', 'cumu_b'), 'CumuB'), (('BNN', 'cumu_-'), 'BNNCumu'), (('LMM-RE', 'cumu_a'), 'LMMCumuA'), (('BNN-RE-batch', 'batch'), 'Batch')]:
    r = pooled.loc[(m, s)]; N[f'auc{tag}'] = f"{r.auc:.2f}"; N[f'sens{tag}'] = f"{r.sens:.2f}"; N[f'spec{tag}'] = f"{r.spec:.2f}"; N[f'brier{tag}'] = f"{r.brier:.3f}"
N['prevalence'] = f"{pooled.iloc[0].prevalence * 100:.0f}"
def avg(m, s, col): return agg[(agg.method == m) & (agg.strategy == s) & (agg.step >= 1)][col].mean()
N['pctNarrowerVsBNN'] = f"{100 * (1 - avg('BNN-RE', 'cumu_a', 'width') / avg('BNN', 'cumu_-', 'width')):.0f}"
N['pctISvsBNN'] = f"{100 * (1 - avg('BNN-RE', 'cumu_a', 'iscore') / avg('BNN', 'cumu_-', 'iscore')):.0f}"
N['pctLMMvsBNNRE'] = f"{100 * (1 - avg('LMM-RE', 'cumu_a', 'width') / avg('BNN-RE', 'cumu_a', 'width')):.1f}"
for s_, tag in [('cumu_a', 'CumuA'), ('cumu_b', 'CumuB'), ('cons_a', 'ConsA'), ('cons_b', 'ConsB')]:
    N[f'mse{tag}D'] = f"{agg[(agg.method == 'BNN-RE') & (agg.strategy == s_) & (agg.step == 4)].mse.values[0]:.3f}"
    N[f'avgMSE{tag}'] = f"{avg('BNN-RE', s_, 'mse'):.3f}"
N['avgMSEBNNCumu'] = f"{avg('BNN', 'cumu_-', 'mse'):.3f}"; N['avgMSELMMCumuA'] = f"{avg('LMM-RE', 'cumu_a', 'mse'):.3f}"; N['avgMSELMMCumuB'] = f"{avg('LMM-RE', 'cumu_b', 'mse'):.3f}"; N['avgMSEBatch'] = f"{avg('BNN-RE-batch', 'batch', 'mse'):.3f}"
N['avgWLMMCumuB'] = f"{avg('LMM-RE', 'cumu_b', 'width'):.2f}"; N['avgCLMMCumuB'] = f"{avg('LMM-RE', 'cumu_b', 'cover'):.3f}"; N['avgISLMMCumuB'] = f"{avg('LMM-RE', 'cumu_b', 'iscore'):.2f}"; N['avgCRPSLMMCumuB'] = f"{avg('LMM-RE', 'cumu_b', 'crps'):.3f}"
blk3 = ex[(ex.step == 2)]; N['aucBlockThreeMin'] = f"{blk3.auc.min():.2f}"; N['aucBlockThreeMax'] = f"{blk3.auc.max():.2f}"
N['sigmaUCumuBD'] = f"{agg[(agg.method == 'BNN-RE') & (agg.strategy == 'cumu_b') & (agg.step == 4)].sigma_u.values[0]:.2f}"
N['sigmaECumuBD'] = f"{agg[(agg.method == 'BNN-RE') & (agg.strategy == 'cumu_b') & (agg.step == 4)].sigma_e.values[0]:.2f}"
for (m, s_), tag in [(('LMM-RE', 'cumu_b'), 'LMMCumuB'), (('BNN-RE', 'cons_b'), 'ConsB')]:
    r = pooled.loc[(m, s_)]; N[f'auc{tag}'] = f"{r.auc:.2f}"; N[f'sens{tag}'] = f"{r.sens:.2f}"; N[f'spec{tag}'] = f"{r.spec:.2f}"; N[f'brier{tag}'] = f"{r.brier:.3f}"
su, se = init.sigma_u.mean(), init.sigma_e.mean()
N['iccPct'] = f"{100 * su**2 / (su**2 + se**2):.0f}"
for m, tag in [('BNN-RE', 'BNNRE'), ('LMM-RE', 'LMM'), ('BNN', 'BNN'), ('BNN-RE-batch', 'Batch')]:
    sub = agg[(agg.method == m) & (agg.step >= 1)]
    N[f'minCover{tag}'] = f"{sub.cover.min():.3f}"; N[f'maxCover{tag}'] = f"{sub.cover.max():.3f}"

with open(f'{P}/numbers_real.tex', 'w') as f:
    for k, v in N.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('macros', len(N))

# ---------------- Figure: boxplots across replicates (width & coverage) -----------------
fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
cols = {'BNN': '#7f7f7f', 'LMM-RE': '#4C78A8', 'BNN-RE': '#E45756', 'BNN-RE-batch': '#54A24B'}
slots = []  # (x position, method, strategy, step, label)
x = 0
slots.append((x, 'initial', 0)); x += 1.5
for k in range(1, 5):
    for mode in ['cons', 'cumu']:
        for var in ['a', 'b']:
            slots.append((x, f'{mode}_{var}', k)); x += 1
        x += 0.3
    x += 0.8
for ax, col in zip(axes, ['width', 'cover']):
    for (xp, strat, k) in slots:
        offs = -0.3
        for m in ['BNN', 'LMM-RE', 'BNN-RE', 'BNN-RE-batch']:
            s = strat if m != 'BNN' else ('initial' if k == 0 else strat.split('_')[0] + '_-')
            if m == 'BNN-RE-batch': s = 'initial' if k == 0 else 'batch'; mm = 'BNN-RE' if k == 0 else m
            else: mm = m
            if m == 'BNN' and strat.endswith('_b') and k > 0: offs += 0.2; continue   # BNN has no (b) variant
            if m == 'BNN-RE-batch' and (k == 0 or strat != 'cons_a' and strat != 'cumu_a'):
                if k > 0: offs += 0.2; continue
            v = df[(df.method == mm) & (df.strategy == s) & (df.step == k)][col].values
            if len(v) == 0: offs += 0.2; continue
            ax.boxplot(v, positions=[xp + offs], widths=0.17, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=cols[m], alpha=0.75, edgecolor='k', linewidth=0.6), medianprops=dict(color='k', linewidth=0.8), whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))
            offs += 0.2
    ax.set_ylabel('Mean 95% interval width' if col == 'width' else 'Empirical coverage')
    if col == 'cover': ax.axhline(0.95, color='k', ls='--', lw=0.8)
labels = ['Initial'] + [f"U{k} {m}{'-carry' if v == 'a' else '-relearn'}" for k in range(1, 5) for m in ['Cons', 'Cumu'] for v in ['a', 'b']]
axes[1].set_xticks([s[0] for s in slots]); axes[1].set_xticklabels(labels, rotation=60, ha='right', fontsize=12)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor=c, alpha=0.75, label=m) for m, c in cols.items()], ncol=4, fontsize=12, loc='upper right', frameon=False)
fig.tight_layout(); fig.savefig(f'{P}/figures/fig_real_boxplots.png', dpi=300); fig.savefig(f'{P}/figures/fig_real_boxplots.pdf')

# ---------------- Figure: forecast vs realised for three sources (cumulative, carry u) -----------------
try:
    srcs_all = get('BNN-RE', 0, 'initial', 'source')[0]
    frames = []
    for k in range(5):
        st = 'initial' if k == 0 else 'cumu_a'
        lo = get('BNN-RE', k, st, 'lo').mean(0); hi = get('BNN-RE', k, st, 'hi').mean(0); mn = get('BNN-RE', k, st, 'mean').mean(0)
        lob = get('BNN', k, 'initial' if k == 0 else 'cumu_-', 'lo').mean(0); hib = get('BNN', k, 'initial' if k == 0 else 'cumu_-', 'hi').mean(0)
        y = get('BNN-RE', k, st, 'y')[0]; src = get('BNN-RE', k, st, 'source')[0]; wk = get('BNN-RE', k, st, 'week')[0]
        frames.append(pd.DataFrame(dict(step=k, lo=lo, hi=hi, mean=mn, lob=lob, hib=hib, y=y, source=src, week=wk)))
    fo = pd.concat(frames); fo.to_csv(f'{P}/tables/forecast_perobs.csv', index=False)
    top = fo.groupby('source').size().sort_values(ascending=False)
    pick = list(top.index[:3])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, s in zip(axes, pick):
        d = fo[fo.source == s].sort_values('week'); xx = np.arange(len(d))
        ax.fill_between(xx, d.lob, d.hib, color='#7f7f7f', alpha=0.25, label='BNN 95% PI', step=None)
        ax.errorbar(xx, d['mean'], yerr=[d['mean'] - d.lo, d.hi - d['mean']], fmt='o', color='#E45756', ms=3.5, capsize=2, lw=1, label='BNN-RE 95% PI (cumulative, carry $u$)')
        ax.plot(xx, d.y, 'k*', ms=7, label='Realised log mortality')
        ax.axhline(TAU_LOG, color='k', ls=':', lw=0.8)
        for b in [5, 10, 15, 20]:
            idx = np.searchsorted(d.week.values, b + 0.5)
            if 0 < idx < len(d): ax.axvline(idx - 0.5, color='grey', ls='--', lw=0.7)
        ax.set_xticks(xx); ax.set_xticklabels([f'w{int(w)}' for w in d.week], rotation=70, fontsize=11)
        ax.set_title(f'Sow source {s} ({len(d)} groups in 2023)', fontsize=13)
    axes[0].set_ylabel('log(% nursery mortality)'); h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=3, frameon=False, fontsize=12)
    fig.tight_layout(rect=[0, 0.12, 1, 1]); fig.savefig(f'{P}/figures/fig_forecast_sources.png', dpi=300); fig.savefig(f'{P}/figures/fig_forecast_sources.pdf')
    print('forecast figure sources', pick)
except Exception as e: print('forecast figure failed', e)

# ---------------- Random-effect ranking vs PRRS status -----------------
try:
    import sys; sys.path.insert(0, '.')
    from prep_data import load_raw
    re_files = sorted(glob.glob('results_real/re_post_seed*.csv')); re = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(re_files)])
    re_m = re.groupby('source').agg(re_mean=('re_mean', 'mean'), re_sd=('re_sd', 'mean'), re_seed_sd=('re_mean', 'std')).reset_index()
    dat = load_raw(); tr = dat[dat.year_num != 2023]
    g = tr.groupby('SowSource').agg(n=('avg_mortality_60days', 'size'), raw_mean=('avg_mortality_60days', 'mean'),
                                     prrs_pos=('PRRSatPlacement', lambda s: (s != 'Negative').mean()), prrs_epi=('PRRSatPlacement', lambda s: (s == 'Epidemic').mean()),
                                     myco_pos=('MYCOatPlacement', lambda s: (s != 'Negative').mean()), multi=('NumberOfSources', 'mean')).reset_index().rename(columns={'SowSource': 'source'})
    re_m = re_m.merge(g, on='source'); re_m.to_csv(f'{P}/tables/random_effects.csv', index=False)
    from scipy.stats import spearmanr
    big = re_m[re_m.n >= 10]
    rho_prrs = spearmanr(big.re_mean, big.prrs_pos); rho_multi = spearmanr(big.re_mean, big.multi); rho_myco = spearmanr(big.re_mean, big.myco_pos)
    neg = big[big.prrs_pos < 0.25].re_mean.mean(); pos = big[big.prrs_pos > 0.75].re_mean.mean()
    with open(f'{P}/numbers_real.tex', 'a') as f:
        f.write(f'\\newcommand{{\\reRange}}{{{re_m.re_mean.min():.2f} to {re_m.re_mean.max():.2f}}}\n')
        f.write(f'\\newcommand{{\\reSD}}{{{re_m.re_mean.std():.2f}}}\n\\newcommand{{\\rhoPRRS}}{{{rho_prrs.statistic:.2f}}}\n\\newcommand{{\\pPRRS}}{{{rho_prrs.pvalue:.3f}}}\n')
        f.write(f'\\newcommand{{\\rhoMyco}}{{{rho_myco.statistic:.2f}}}\n\\newcommand{{\\rhoMulti}}{{{rho_multi.statistic:.2f}}}\n')
        f.write(f'\\newcommand{{\\reNegPRRS}}{{{neg:.2f}}}\n\\newcommand{{\\rePosPRRS}}{{{pos:.2f}}}\n\\newcommand{{\\nBigSources}}{{{len(big)}}}\n')
        f.write(f'\\newcommand{{\\reSeedSD}}{{{re_m.re_seed_sd.mean():.3f}}}\n')
        f.write(f'\\newcommand{{\\reLowPct}}{{{100 * (1 - np.exp(re_m.re_mean.min())):.0f}}}\n\\newcommand{{\\reHighPct}}{{{100 * (np.exp(re_m.re_mean.max()) - 1):.0f}}}\n')
        f.write(f'\\newcommand{{\\rhoMultiP}}{{{rho_multi.pvalue:.3f}}}\n')
    print('RE ranking: rho PRRS', rho_prrs, 'myco', rho_myco, 'multi', rho_multi, 'neg vs pos', neg, pos)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    o = re_m.sort_values('re_mean').reset_index(drop=True)
    colors = plt.cm.RdYlBu_r(o.prrs_pos.values)
    ax.bar(np.arange(len(o)), o.re_mean, yerr=o.re_sd, color=colors, edgecolor='none', error_kw=dict(lw=0.4, alpha=0.6))
    ax.set_xlabel('Sow source (ranked by posterior mean random effect)'); ax.set_ylabel('Random effect $u_i$ (log scale)'); ax.set_xticks([])
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1)); sm.set_array([]); cb = fig.colorbar(sm, ax=ax, pad=0.01); cb.set_label('Share of groups PRRS-positive at placement')
    fig.tight_layout(); fig.savefig(f'{P}/figures/fig_random_effects.png', dpi=300); fig.savefig(f'{P}/figures/fig_random_effects.pdf')
except Exception as e: print('RE figure failed', e)
print('done')
