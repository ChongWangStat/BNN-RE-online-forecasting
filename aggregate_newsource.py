import glob, numpy as np, pandas as pd, matplotlib
# FRONTIERS-FONTS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})  # >= 8 pt at 180 mm final width
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
P = os.environ.get('BNNRE_OUT', 'outputs'); os.makedirs(f'{P}/figures', exist_ok=True); os.makedirs(f'{P}/tables', exist_ok=True)
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results_newsource/newsource_seed*.csv'))]); R = df.seed.nunique(); print('reps', R, 'rows', len(df))
bins = [-1, 0, 4, 9, 19, 1000]; labels = ['0', '1--4', '5--9', '10--19', '$\\geq$20']
df['bin'] = pd.cut(df.n_seen, bins=bins, labels=labels)
df['key'] = df.method + '|' + df.strategy
# average within seed first (so MC SE is across VI re-initialisations), then across seeds
per = df.groupby(['seed', 'method', 'strategy', 'bin'], observed=True).agg(width=('width', 'mean'), cover=('cover', 'mean'), iscore=('iscore', 'mean'), n=('width', 'size')).reset_index()
agg = per.groupby(['method', 'strategy', 'bin'], observed=True).agg(width=('width', 'mean'), width_se=('width', lambda v: v.std(ddof=1) / np.sqrt(len(v))), cover=('cover', 'mean'), cover_se=('cover', lambda v: v.std(ddof=1) / np.sqrt(len(v))), iscore=('iscore', 'mean'), n=('n', 'mean')).reset_index()
agg.to_csv(f'{P}/tables/newsource_summary.csv', index=False); pd.set_option('display.width', 250); print(agg.round(3).to_string())
# pool strategies: combine the 'initial' step (n_seen = 0) into each strategy's curve
rows = {}
for method, strat in [('BNN', 'cumu_-'), ('BNN', 'cons_-'), ('LMM-RE', 'cumu_a'), ('BNN-RE', 'cons_a'), ('BNN-RE', 'cons_b'), ('BNN-RE', 'cumu_a'), ('BNN-RE', 'cumu_b')]:
    d = df[(df.method == method) & ((df.strategy == strat) | (df.strategy == 'initial'))]
    p = d.groupby(['seed', 'bin'], observed=True).agg(width=('width', 'mean'), cover=('cover', 'mean'), iscore=('iscore', 'mean'), n=('width', 'size')).reset_index()
    a = p.groupby('bin', observed=True).agg(width=('width', 'mean'), width_se=('width', lambda v: v.std(ddof=1) / np.sqrt(len(v))), cover=('cover', 'mean'), cover_se=('cover', lambda v: v.std(ddof=1) / np.sqrt(len(v))), iscore=('iscore', 'mean'), n=('n', 'mean'))
    rows[(method, strat)] = a
lab = {('BNN', 'cumu_-'): 'BNN, cumulative', ('BNN', 'cons_-'): 'BNN, consecutive', ('LMM-RE', 'cumu_a'): 'LMM-RE, cumu., carry $\\bm u$', ('BNN-RE', 'cons_a'): 'BNN-RE, cons., carry $\\bm u$', ('BNN-RE', 'cons_b'): 'BNN-RE, cons., re-learn',
       ('BNN-RE', 'cumu_a'): 'BNN-RE, cumu., carry $\\bm u$', ('BNN-RE', 'cumu_b'): 'BNN-RE, cumu., re-learn'}
L = [r'\begin{tabular}{l ' + 'c' * len(labels) + ' ' + 'c' * len(labels) + '}', r'\hline', r' & \multicolumn{5}{c}{Mean width by groups already observed from the source} & \multicolumn{5}{c}{Coverage} \\', r'\cline{2-6}\cline{7-11}',
     'Method, strategy & ' + ' & '.join(labels) + ' & ' + ' & '.join(labels) + r' \\', r'\hline']
for key, a in rows.items():
    w = [f"{a.loc[b, 'width']:.2f}" if b in a.index else '--' for b in labels]; c = [f"{a.loc[b, 'cover']:.3f}" if b in a.index else '--' for b in labels]
    L.append(f"{lab[key]} & " + ' & '.join(w) + ' & ' + ' & '.join(c) + r' \\')
L += [r'\hline', r'\end{tabular}']
open(f'{P}/tables/table_newsource.tex', 'w').write('\n'.join(L))
nobs = rows[('BNN-RE', 'cumu_a')]['n']
with open(f'{P}/numbers_newsource.tex', 'w') as f:
    f.write(f'\\newcommand{{\\nsReps}}{{{R}}}\n')
    for key, tag in [(('BNN-RE', 'cumu_a'), 'CumuA'), (('BNN-RE', 'cumu_b'), 'CumuB'), (('BNN-RE', 'cons_a'), 'ConsA'), (('BNN-RE', 'cons_b'), 'ConsB'), (('BNN', 'cumu_-'), 'BNN'), (('LMM-RE', 'cumu_a'), 'LMM')]:
        a = rows[key]
        for b, bt in zip(labels, ['Zero', 'Few', 'Five', 'Ten', 'Twenty']):
            if b in a.index: f.write(f'\\newcommand{{\\nsW{tag}{bt}}}{{{a.loc[b, "width"]:.2f}}}\n\\newcommand{{\\nsC{tag}{bt}}}{{{a.loc[b, "cover"]:.3f}}}\n')
    for b, bt in zip(labels, ['Zero', 'Few', 'Five', 'Ten', 'Twenty']):
        if b in nobs.index: f.write(f'\\newcommand{{\\nsN{bt}}}{{{nobs.loc[b]:.0f}}}\n')

# ---------------- per-block table for the supplement -----------------
blk = df.groupby(['seed', 'method', 'strategy', 'step']).agg(width=('width', 'mean'), cover=('cover', 'mean'), iscore=('iscore', 'mean')).reset_index()
blk = blk.groupby(['method', 'strategy', 'step']).agg(width=('width', 'mean'), cover=('cover', 'mean'), iscore=('iscore', 'mean')).reset_index()
blk.to_csv(f'{P}/tables/newsource_blocks.csv', index=False)
order = [('BNN', 'initial'), ('BNN', 'cons_-'), ('BNN', 'cumu_-'), ('LMM-RE', 'initial'), ('LMM-RE', 'cons_a'), ('LMM-RE', 'cons_b'), ('LMM-RE', 'cumu_a'), ('LMM-RE', 'cumu_b'), ('BNN-RE', 'initial'), ('BNN-RE', 'cons_a'), ('BNN-RE', 'cons_b'), ('BNN-RE', 'cumu_a'), ('BNN-RE', 'cumu_b')]
lab2 = dict(lab); lab2.update({('BNN', 'initial'): 'BNN, initial model', ('LMM-RE', 'initial'): 'LMM-RE, initial model', ('BNN-RE', 'initial'): 'BNN-RE, initial model', ('LMM-RE', 'cons_a'): 'LMM-RE, cons., carry $\\bm u$', ('LMM-RE', 'cons_b'): 'LMM-RE, cons., re-learn', ('LMM-RE', 'cumu_b'): 'LMM-RE, cumu., re-learn'})
L = [r'\begin{tabular}{l cccc cccc cccc}', r'\hline', r' & \multicolumn{4}{c}{Width} & \multicolumn{4}{c}{Coverage} & \multicolumn{4}{c}{Interval score} \\', r'\cline{2-5}\cline{6-9}\cline{10-13}',
     r'Method, strategy & Block 2 & Block 3 & Block 4 & Block 5 & Block 2 & Block 3 & Block 4 & Block 5 & Block 2 & Block 3 & Block 4 & Block 5 \\', r'\hline']
for m, st in order:
    b = blk[(blk.method == m) & (blk.strategy == st)]
    if len(b) == 0: continue
    if st == 'initial':
        r = b.iloc[0]; L.append(f"{lab2[(m, st)]} (block 1) & {r.width:.2f} & & & & {r.cover:.3f} & & & & {r.iscore:.2f} & & & \\\\"); continue
    cells = lambda col, nd: ' & '.join(f"{b[b.step == k][col].values[0]:.{nd}f}" if (b.step == k).any() else '--' for k in range(1, 5))
    L.append(f"{lab2[(m, st)]} & {cells('width', 2)} & {cells('cover', 3)} & {cells('iscore', 2)} \\\\")
L += [r'\hline', r'\end{tabular}']
open(f'{P}/tables/table_newsource_blocks.tex', 'w').write('\n'.join(L))
print('block table written')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
labfig = {k: v.replace('$\\bm u$', 'u') for k, v in lab.items()}
sty = {('BNN', 'cumu_-'): ('#7f7f7f', '-', 'o'), ('LMM-RE', 'cumu_a'): ('#4C78A8', '-', 's'), ('BNN-RE', 'cons_a'): ('#E45756', '--', '^'), ('BNN-RE', 'cons_b'): ('#E45756', ':', 'v'), ('BNN-RE', 'cumu_a'): ('#E45756', '-', 'o'), ('BNN-RE', 'cumu_b'): ('#B2182B', '-.', 'D')}
for key, (c, ls, mk) in sty.items():
    a = rows[key]; xs = np.arange(len(labels))
    axes[0].errorbar(xs, [a.loc[b, 'width'] if b in a.index else np.nan for b in labels], yerr=[a.loc[b, 'width_se'] if b in a.index else 0 for b in labels], color=c, ls=ls, marker=mk, ms=4, capsize=2, label=labfig[key])
    axes[1].errorbar(xs, [a.loc[b, 'cover'] if b in a.index else np.nan for b in labels], yerr=[a.loc[b, 'cover_se'] if b in a.index else 0 for b in labels], color=c, ls=ls, marker=mk, ms=4, capsize=2)
for ax in axes: ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels([l.replace('$\\geq$', '≥') for l in labels]); ax.set_xlabel('Groups already observed from the new sow source')
axes[0].set_ylabel('Mean 95% interval width'); axes[1].set_ylabel('Empirical coverage'); axes[1].axhline(0.95, color='k', ls='--', lw=0.8)
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=3, frameon=False, fontsize=12); fig.tight_layout(rect=[0, 0.18, 1, 1]); fig.savefig(f'{P}/figures/fig_newsource.png', dpi=300); fig.savefig(f'{P}/figures/fig_newsource.pdf'); print('done')
