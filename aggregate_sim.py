import glob, numpy as np, pandas as pd, matplotlib
# FRONTIERS-FONTS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})  # >= 8 pt at 180 mm final width
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
P = os.environ.get('BNNRE_OUT', 'outputs'); os.makedirs(f'{P}/figures', exist_ok=True); os.makedirs(f'{P}/tables', exist_ok=True)
files = sorted(glob.glob('results_sim/sim_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files]); print(df.groupby(['dgp', 'scenario']).seed.nunique())
agg = df.groupby(['dgp', 'scenario', 'method', 'strategy', 'step']).agg(width=('width', 'mean'), width_se=('width', lambda v: v.std(ddof=1) / np.sqrt(len(v))), cover=('cover', 'mean'),
      cover_se=('cover', lambda v: v.std(ddof=1) / np.sqrt(len(v))), iscore=('iscore', 'mean'), crps=('crps', 'mean'), sigma_e=('sigma_e', 'mean'), sigma_u=('sigma_u', 'mean'), n=('seed', 'size')).reset_index()
agg.to_csv(f'{P}/tables/sim_summary.csv', index=False)
pd.set_option('display.width', 250); print(agg.round(3).to_string())
cols = {'BNN': '#7f7f7f', 'LMM-RE': '#4C78A8', 'BNN-RE': '#E45756', 'BNN-RE-batch': '#54A24B'}
names = {'linear': 'Linear $f$', 'nonlinear': 'Nonlinear $f$', 'nn': 'Single-layer network $f$'}
for scen in ['shift', 'noshift']:
    sub0 = df[df.scenario == scen]
    if len(sub0) == 0: continue
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey='row')
    for ax in axes.ravel(): ax.tick_params(labelsize=15)
    for j, dgp in enumerate(['linear', 'nonlinear', 'nn']):
        d = sub0[sub0.dgp == dgp]
        if len(d) == 0: continue
        slots = [(0, 'initial', 0)]; x = 1.5
        for k in range(1, 5):
            for mode in ['cons', 'cumu']:
                for var in ['a', 'b']: slots.append((x, f'{mode}_{var}', k)); x += 1
                x += 0.3
            x += 0.8
        for i, col in enumerate(['width', 'cover']):
            ax = axes[i, j]
            for (xp, strat, k) in slots:
                offs = -0.3
                for m in ['BNN', 'LMM-RE', 'BNN-RE', 'BNN-RE-batch']:
                    if m == 'BNN': s = 'initial' if k == 0 else strat.split('_')[0] + '_-'; mm = m
                    elif m == 'BNN-RE-batch': s = 'initial' if k == 0 else 'batch'; mm = 'BNN-RE' if k == 0 else m
                    else: s = strat; mm = m
                    if k > 0 and m == 'BNN' and strat.endswith('_b'): offs += 0.2; continue
                    if k > 0 and m == 'BNN-RE-batch' and strat.endswith('_b'): offs += 0.2; continue
                    v = d[(d.method == mm) & (d.strategy == s) & (d.step == k)][col].values
                    if len(v) == 0: offs += 0.2; continue
                    ax.boxplot(v, positions=[xp + offs], widths=0.17, patch_artist=True, showfliers=False, boxprops=dict(facecolor=cols[m], alpha=0.75, edgecolor='k', linewidth=0.5),
                               medianprops=dict(color='k', linewidth=0.8), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
                    offs += 0.2
            if col == 'cover': ax.axhline(0.95, color='k', ls='--', lw=0.8)
            if j == 0: ax.set_ylabel('Mean 95% interval width' if col == 'width' else 'Empirical coverage')
            if i == 0: ax.set_title(names[dgp], fontsize=13)
        labels = ['Initial'] + [f"U{k} {m}{'-carry' if v == 'a' else '-relearn'}" for k in range(1, 5) for m in ['Cons', 'Cumu'] for v in ['a', 'b']]
        axes[1, j].set_xticks([s[0] for s in slots]); axes[1, j].set_xticklabels(labels, rotation=70, ha='right', fontsize=11)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, alpha=0.75, label=m) for m, c in cols.items()], ncol=4, fontsize=14, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f'{P}/figures/fig_sim_{scen}.png', dpi=300); fig.savefig(f'{P}/figures/fig_sim_{scen}.pdf'); print('saved', scen)
# numbers
N = {}
for scen in ['shift', 'noshift']:
    for dgp, tag in [('linear', 'Lin'), ('nonlinear', 'Nonlin'), ('nn', 'NN')]:
        a = agg[(agg.scenario == scen) & (agg.dgp == dgp)]
        if len(a) == 0: continue
        T = ('S' if scen == 'shift' else 'N') + tag
        def m(method, strat, col, nd=2):
            s = a[(a.method == method) & (a.strategy == strat) & (a.step >= 1)]; return f"{s[col].mean():.{nd}f}" if len(s) else '--'
        N[f'simReps{T}'] = str(int(a.n.max()))
        for method, strat, mt in [('BNN-RE', 'cons_a', 'ConsA'), ('BNN-RE', 'cons_b', 'ConsB'), ('BNN-RE', 'cumu_a', 'CumuA'), ('BNN-RE', 'cumu_b', 'CumuB'), ('BNN', 'cons_-', 'BNNCons'), ('BNN', 'cumu_-', 'BNNCumu'), ('LMM-RE', 'cumu_a', 'LMMCumuA'), ('LMM-RE', 'cons_a', 'LMMConsA'), ('LMM-RE', 'cumu_b', 'LMMCumuB'), ('BNN-RE-batch', 'batch', 'Batch')]:
            N[f'simW{T}{mt}'] = m(method, strat, 'width'); N[f'simC{T}{mt}'] = m(method, strat, 'cover', 3); N[f'simIS{T}{mt}'] = m(method, strat, 'iscore')
        i0 = a[(a.step == 0)]
        for method, mt in [('BNN-RE', 'BNNRE'), ('BNN', 'BNN'), ('LMM-RE', 'LMM')]:
            r = i0[i0.method == method]
            if len(r): N[f'simWinit{T}{mt}'] = f"{r.width.values[0]:.2f}"; N[f'simCinit{T}{mt}'] = f"{r.cover.values[0]:.3f}"
        r = a[(a.method == 'BNN-RE') & (a.strategy == 'cumu_b') & (a.step == 4)]
        if len(r): N[f'simSigU{T}CumuB'] = f"{r.sigma_u.values[0]:.2f}"
        r = a[(a.method == 'BNN-RE') & (a.strategy == 'initial')]
        if len(r): N[f'simSigU{T}Init'] = f"{r.sigma_u.values[0]:.2f}"; N[f'simSigE{T}Init'] = f"{r.sigma_e.values[0]:.2f}"
        r = a[(a.method == 'BNN-RE') & (a.strategy == 'cons_b') & (a.step == 4)]
        if len(r): N[f'simSigU{T}ConsB'] = f"{r.sigma_u.values[0]:.2f}"; N[f'simSigE{T}ConsB'] = f"{r.sigma_e.values[0]:.2f}"
with open(f'{P}/numbers_sim.tex', 'w') as f:
    for k, v in N.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
# compact LaTeX table: average over updates 1-4 of width / coverage / interval score per scenario x dgp x method-strategy
L = [r'\begin{tabular}{ll l ccc ccc}', r'\hline', r'Scenario & $f$ & Method, strategy & \multicolumn{3}{c}{Initial model} & \multicolumn{3}{c}{Mean over updates 1--4} \\', r'\cline{4-6}\cline{7-9}',
     r' & & & Width & Coverage & IS & Width & Coverage & IS \\', r'\hline']
lab = {('BNN', 'cons_-'): 'BNN, consecutive', ('BNN', 'cumu_-'): 'BNN, cumulative', ('LMM-RE', 'cons_a'): 'LMM-RE, cons., carry $\\bm u$', ('LMM-RE', 'cumu_a'): 'LMM-RE, cumu., carry $\\bm u$', ('LMM-RE', 'cumu_b'): 'LMM-RE, cumu., re-learn',
       ('BNN-RE', 'cons_a'): 'BNN-RE, cons., carry $\\bm u$', ('BNN-RE', 'cons_b'): 'BNN-RE, cons., re-learn', ('BNN-RE', 'cumu_a'): 'BNN-RE, cumu., carry $\\bm u$', ('BNN-RE', 'cumu_b'): 'BNN-RE, cumu., re-learn', ('BNN-RE-batch', 'batch'): 'BNN-RE, batch retrain'}
for scen in ['shift', 'noshift']:
    for dgp in ['linear', 'nonlinear', 'nn']:
        a = agg[(agg.scenario == scen) & (agg.dgp == dgp)]
        if len(a) == 0: continue
        first = True
        for (method, strat), name in lab.items():
            s = a[(a.method == method) & (a.strategy == strat) & (a.step >= 1)]
            if len(s) == 0: continue
            im = 'BNN-RE' if method == 'BNN-RE-batch' else method
            i0 = a[(a.method == im) & (a.step == 0)].iloc[0]
            L.append(f"{('Variance shift' if scen == 'shift' else 'No shift') if first else ''} & {names[dgp] if first else ''} & {name} & {i0.width:.2f} & {i0.cover:.3f} & {i0.iscore:.2f} & {s.width.mean():.2f} & {s.cover.mean():.3f} & {s.iscore.mean():.2f} \\\\")
            first = False
        L.append(r'\hline')
L.append(r'\end{tabular}')
open(f'{P}/tables/table_sim.tex', 'w').write('\n'.join(L)); print('done')

# ---------------- per-functional-form figures in the original two-panel style (variance-shift scenario) -----------------
for scen in ['shift', 'noshift']:
    for dgp in ['linear', 'nonlinear', 'nn']:
        d = df[(df.scenario == scen) & (df.dgp == dgp)]
        if len(d) == 0: continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
        dsc = df[df.scenario == scen]
        ylims = {'width': (dsc.width.quantile(0.002) - 0.03, dsc.width.quantile(0.998) + 0.03), 'cover': (max(0.8, dsc.cover.min() - 0.01), 1.005)}   # shared across the three functional forms
        slots = [(0, 'initial', 0)]; x = 1.5
        for k in range(1, 5):
            for mode in ['cons', 'cumu']:
                for var in ['a', 'b']: slots.append((x, f'{mode}_{var}', k)); x += 1
                x += 0.3
            x += 0.8
        for ax, col in zip(axes, ['width', 'cover']):
            for (xp, strat, k) in slots:
                offs = -0.3
                for m in ['BNN', 'LMM-RE', 'BNN-RE', 'BNN-RE-batch']:
                    if m == 'BNN': s_ = 'initial' if k == 0 else strat.split('_')[0] + '_-'; mm = m
                    elif m == 'BNN-RE-batch': s_ = 'initial' if k == 0 else 'batch'; mm = 'BNN-RE' if k == 0 else m
                    else: s_ = strat; mm = m
                    if k > 0 and m in ('BNN', 'BNN-RE-batch') and strat.endswith('_b'): offs += 0.2; continue
                    v = d[(d.method == mm) & (d.strategy == s_) & (d.step == k)][col].values
                    if len(v) == 0: offs += 0.2; continue
                    ax.boxplot(v, positions=[xp + offs], widths=0.17, patch_artist=True, showfliers=False, boxprops=dict(facecolor=cols[m], alpha=0.75, edgecolor='k', linewidth=0.5),
                               medianprops=dict(color='k', linewidth=0.8), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
                    offs += 0.2
            if col == 'cover': ax.axhline(0.95, color='k', ls='--', lw=0.8)
            ax.set_ylim(*ylims[col])
            ax.set_ylabel('Average interval width' if col == 'width' else 'Average coverage rate')
        labels = ['Initial'] + [f"U{k} {m}{'-carry' if v == 'a' else '-relearn'}" for k in range(1, 5) for m in ['Cons', 'Cumu'] for v in ['a', 'b']]
        axes[1].set_xticks([s_[0] for s_ in slots]); axes[1].set_xticklabels(labels, rotation=60, ha='right', fontsize=12)
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor=c, alpha=0.75, label=m) for m, c in cols.items()], ncol=4, fontsize=12, frameon=False, loc='upper right')
        fig.tight_layout(); fig.savefig(f'{P}/figures/fig_sim_{scen}_{dgp}.png', dpi=300); fig.savefig(f'{P}/figures/fig_sim_{scen}_{dgp}.pdf'); plt.close(fig)
        print('saved per-dgp figure', scen, dgp)
