"""Build all tables, figures and key numbers for the manuscript from the saved run outputs.

Methods reported:  BNN; BNN-RE (carry-forward update); BNN-RE-T (tempered sigma_e prior); BNN-RE-C (BNN-RE with
conformal calibration of the interval width on the test blocks already observed, computed post hoc from the saved
per-observation predictions).  Usage: python make_results.py OUT_REAL OUT_REAL_T OUT_SIM OUT_SIM_LONG DEST TIFFDIR"""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from conformal import calibrate_sequence
out_real, out_t, out_sim, out_long, dest, tiffdir = sys.argv[1:7]
os.makedirs(dest, exist_ok=True); os.makedirs(tiffdir, exist_ok=True); os.makedirs(f'{dest}/../figures', exist_ok=True)
FIGDIR = os.path.normpath(f'{dest}/../figures')
METHODS = ['BNN', 'BNN-RE', 'BNN-RE-T', 'BNN-RE-C']
ALPHA_CAL = float(os.environ.get('ALPHA_CAL', '0.04'))   # calibration level of BNN-RE-C (0.04 = one-point margin over the nominal 0.05)
HEAD = r'\textbf{BNN} & \textbf{BNN-RE} & \textbf{BNN-RE-T} & \textbf{BNN-RE-C}'
key = {}

def fix_update1(df):
    df = df[~((df.strategy == 'cumu') & (df.eval_block == 2))].copy()
    c = df[(df.strategy == 'cons') & (df.eval_block == 2)].copy(); c['strategy'] = 'cumu'
    return pd.concat([df, c], ignore_index=True)

# ------------------------------------------------------------------ field data: per seed, per method
rows = []
for f in sorted(glob.glob(f'{out_real}/perobs_seed*.npz')):
    seed = int(f.split('seed')[1].split('.')[0]); z = np.load(f); zt = np.load(f'{out_t}/perobs_seed{seed}.npz')
    for strat in ('cons', 'cumu'):
        seqs = {'BNN': [z['BNN|initial|1']] + [z[f'BNN|{strat}|{k}'] for k in (2, 3, 4, 5)],
                'BNN-RE': [z['BNN-RE|initial|1']] + [z[f'BNN-RE(a)|{strat}|{k}'] for k in (2, 3, 4, 5)],
                'BNN-RE-T': [zt['initial|1']] + [zt[f'{strat}|{k}'] for k in (2, 3, 4, 5)]}
        for m, blocks in seqs.items():
            for k, arr in enumerate(blocks):
                lo, hi, _, y = arr
                rows.append(dict(seed=seed, method=m, strategy=strat, eval_block=k + 1, width=float(np.mean(hi - lo)), cover=float(np.mean((lo < y) & (y < hi)))))
        for k, r in enumerate(calibrate_sequence(seqs['BNN-RE'], alpha=ALPHA_CAL)):
            rows.append(dict(seed=seed, method='BNN-RE-C', strategy=strat, eval_block=k + 1, width=r['width'], cover=r['cover'], q=r['q']))
F = pd.DataFrame(rows); key['n_seeds'] = int(F.seed.nunique())
MF = F.groupby(['method', 'strategy', 'eval_block'])[['width', 'cover']].mean()

def cell(M, m, s, k, col, fmt):
    try: return fmt % M.loc[(m, s, k), col]
    except KeyError: return '--'

def table_rows(M, meths, nupd=4):
    lines = [r'\multicolumn{2}{c|}{\textbf{Initial}} & ' + ' & '.join(cell(M, m, 'cons', 1, 'width', '%.2f') for m in meths) + ' & ' + ' & '.join(cell(M, m, 'cons', 1, 'cover', '%.3f') for m in meths) + r' \\', r'\hline',
             r'\multicolumn{2}{c|}{\textbf{Update \#1}} & ' + ' & '.join(cell(M, m, 'cons', 2, 'width', '%.2f') for m in meths) + ' & ' + ' & '.join(cell(M, m, 'cons', 2, 'cover', '%.3f') for m in meths) + r' \\']
    for u in range(2, nupd + 1):
        lines.append(r'\hline')
        for j, s in enumerate(('cons', 'cumu')):
            head = r'\multirow{2}{*}{\textbf{Update \#%d}} & Cons.' % u if j == 0 else ' & Cumu.'
            lines.append(head + ' & ' + ' & '.join(cell(M, m, s, u + 1, 'width', '%.2f') for m in meths) + ' & ' + ' & '.join(cell(M, m, s, u + 1, 'cover', '%.3f') for m in meths) + r' \\')
    return lines

def wrap(lines, first_col='cc|cccc|cccc'):
    return '\n'.join([r'\begin{tabular}{%s}' % first_col, r'\hline',
                      r'&& \multicolumn{4}{c|}{\textbf{Average Interval Width}} & \multicolumn{4}{c}{\textbf{Average Interval Coverage Rate}} \\', r'\hline',
                      '&& ' + HEAD + ' & ' + HEAD + r' \\', r'\hline'] + lines + [r'\hline', r'\end{tabular}']) + '\n'
open(f'{dest}/app_res_table.tex', 'w').write(wrap(table_rows(MF, METHODS)))

# key numbers (field)
def mean_upd(M, m, s, col, ks): return float(np.mean([M.loc[(m, s, k), col] for k in ks if (m, s, k) in M.index]))
for m in METHODS:
    for s in ('cons', 'cumu'):
        key[f'field|{m}|{s}|width'] = round(mean_upd(MF, m, s, 'width', (2, 3, 4, 5)), 3); key[f'field|{m}|{s}|cover'] = round(mean_upd(MF, m, s, 'cover', (2, 3, 4, 5)), 3)
key['field_min_cover'] = {m: round(float(MF.xs(m, level='method').cover.min()), 3) for m in METHODS}
key['field_RE_narrower_than_BNN_all'] = bool(all(MF.loc[('BNN-RE', s, k), 'width'] < MF.loc[('BNN', s, k), 'width'] for (m, s, k) in MF.index if m == 'BNN-RE'))
key['field_C_vs_RE_pct_by_block'] = {f'{s}{k}': round(100 * (MF.loc[('BNN-RE-C', s, k), 'width'] / MF.loc[('BNN-RE', s, k), 'width'] - 1), 1) for (m, s, k) in MF.index if m == 'BNN-RE-C' and k >= 2}
key['field_T_vs_RE_pct_by_block'] = {f'{s}{k}': round(100 * (MF.loc[('BNN-RE-T', s, k), 'width'] / MF.loc[('BNN-RE', s, k), 'width'] - 1), 1) for (m, s, k) in MF.index if m == 'BNN-RE-T' and k >= 2}
_sh = F.groupby(['method', 'strategy', 'eval_block']).cover.apply(lambda v: float((v < 0.95).mean())).round(2)
key['field_share_seeds_below_095'] = {f'{m}|{s}': {int(k): float(v) for k, v in _sh.loc[(m, s)].items()} for (m, s) in {(m, s) for (m, s, k) in _sh.index}}

# ------------------------------------------------------------------ simulations (5 blocks) and long horizon
def sim_frame(out, forms, kb):
    rows = []
    for form in forms:
        files = sorted(glob.glob(f'{out}/sim_{form}{"" if kb == 5 else f"_K{kb}"}_rep*.csv'))
        if not files: continue
        d = fix_update1(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        d['method'] = d.method.replace({'BNN-RE': 'BNN-RE', 'BNN-RE(a)': 'BNN-RE', 'BNN-RE(c)': 'BNN-RE-T'})
        d = d[d.method.isin(['BNN', 'BNN-RE', 'BNN-RE-T'])]
        d.loc[d.strategy == 'initial', 'strategy'] = 'cons'          # initial fit is shared by both schedules
        t0 = d[(d.method == 'BNN-RE') & (d.eval_block == 1)].copy(); t0['method'] = 'BNN-RE-T'   # tempering starts at the first update
        d = pd.concat([d, t0], ignore_index=True)
        rows.append(d[['form', 'rep', 'method', 'strategy', 'eval_block', 'width', 'cover']])
        z = np.load(f'{out}/perobs_{form}{"" if kb == 5 else f"_K{kb}"}.npz')
        R = z['BNN|initial|1'].shape[0]
        for strat in [st for st in ('cons', 'cumu') if f'BNN-RE(a)|{st}|{kb}' in z]:
            for r in range(R):
                blocks = [z['BNN-RE|initial|1'][r]] + [z[f'BNN-RE(a)|{strat}|{k}'][r] for k in range(2, kb + 1)]
                for k, res in enumerate(calibrate_sequence(blocks, alpha=ALPHA_CAL)):
                    rows.append(pd.DataFrame([dict(form=form, rep=r, method='BNN-RE-C', strategy=strat, eval_block=k + 1, width=res['width'], cover=res['cover'])]))
    return pd.concat(rows, ignore_index=True) if rows else None

FORMS = [('linear', 'Linear'), ('nonlinear', 'Nonlinear'), ('nn', 'NN')]
S5 = sim_frame(out_sim, [f for f, _ in FORMS], 5)
if S5 is not None:
    key['sim_reps'] = S5.groupby('form').rep.nunique().to_dict()
    MS = S5.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    lines = []
    for f, name in FORMS:
        if f not in S5.form.unique(): continue
        M = MS.xs(f, level='form')
        lines.append(r'\multirow{8}{*}{%s} & Initial & ' % name + ' & '.join(cell(M, m, 'cons', 1, 'width', '%.2f') for m in METHODS) + ' & ' + ' & '.join(cell(M, m, 'cons', 1, 'cover', '%.3f') for m in METHODS) + r' \\')
        lines.append(r' & Update \#1 & ' + ' & '.join(cell(M, m, 'cons', 2, 'width', '%.2f') for m in METHODS) + ' & ' + ' & '.join(cell(M, m, 'cons', 2, 'cover', '%.3f') for m in METHODS) + r' \\')
        for u in (2, 3, 4):
            for s, lab in (('cons', 'Cons.'), ('cumu', 'Cumu.')):
                lines.append(r' & Update \#%d %s & ' % (u, lab) + ' & '.join(cell(M, m, s, u + 1, 'width', '%.2f') for m in METHODS) + ' & ' + ' & '.join(cell(M, m, s, u + 1, 'cover', '%.3f') for m in METHODS) + r' \\')
        lines.append(r'\hline')
    body = '\n'.join([r'\begin{tabular}{cl|cccc|cccc}', r'\hline', r'&& \multicolumn{4}{c|}{\textbf{Average Interval Width}} & \multicolumn{4}{c}{\textbf{Average Interval Coverage Rate}} \\', r'\hline',
                      r'\textbf{Form} & & ' + HEAD + ' & ' + HEAD + r' \\', r'\hline'] + lines + [r'\end{tabular}']) + '\n'
    open(f'{dest}/sim_res_table.tex', 'w').write(body)
    key['sim_min_cover'] = {m: round(float(MS.xs(m, level='method').cover.min()), 3) for m in METHODS}
    key['sim_RE_narrower_all'] = bool(all(MS.loc[(f, 'BNN-RE', s, k), 'width'] < MS.loc[(f, 'BNN', s, k), 'width'] for (f, m, s, k) in MS.index if m == 'BNN-RE'))
    for mm in ('BNN-RE-T', 'BNN-RE-C'):
        key[f'sim_{mm}_vs_RE_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (MS.loc[(f, mm, s, k), 'width'] / MS.loc[(f, 'BNN-RE', s, k), 'width'] - 1), 1) for (f, m, s, k) in MS.index if m == mm and k >= 2}
        key[f'sim_{mm}_vs_BNN_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (MS.loc[(f, mm, s, k), 'width'] / MS.loc[(f, 'BNN', s, k), 'width'] - 1), 1) for (f, m, s, k) in MS.index if m == mm and k >= 2}
    key['sim_RE_vs_BNN_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (MS.loc[(f, 'BNN-RE', s, k), 'width'] / MS.loc[(f, 'BNN', s, k), 'width'] - 1), 1) for (f, m, s, k) in MS.index if m == 'BNN-RE'}
    for f, _ in FORMS:
        if f in S5.form.unique():
            for m in METHODS:
                key[f'sim|{f}|{m}|cumu|width'] = round(mean_upd(MS.xs(f, level='form'), m, 'cumu', 'width', (2, 3, 4, 5)), 3)
                key[f'sim|{f}|{m}|cumu|cover'] = round(mean_upd(MS.xs(f, level='form'), m, 'cumu', 'cover', (2, 3, 4, 5)), 3)

SL = sim_frame(out_long, [f for f, _ in FORMS], 21)
if SL is not None:
    key['long_reps'] = SL.groupby('form').rep.nunique().to_dict()
    ML = SL.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    REPORT = [(1, 'Initial', 'cons')] + [(u + 1, r'Update \#%d' % u, 'cons') for u in (1, 5, 10, 15, 20)]
    lines = []
    for f, name in FORMS:
        if f not in SL.form.unique(): continue
        M = ML.xs(f, level='form')
        for j, (k, lab, s) in enumerate(REPORT):
            head = (r'\multirow{%d}{*}{%s} & ' % (len(REPORT), name)) if j == 0 else ' & '
            lines.append(head + lab + ' & ' + ' & '.join(cell(M, m, s, k, 'width', '%.2f') for m in METHODS) + ' & ' + ' & '.join(cell(M, m, s, k, 'cover', '%.3f') for m in METHODS) + r' \\')
        lines.append(r'\hline')
    body = '\n'.join([r'\begin{tabular}{cl|cccc|cccc}', r'\hline', r'&& \multicolumn{4}{c|}{\textbf{Average Interval Width}} & \multicolumn{4}{c}{\textbf{Average Interval Coverage Rate}} \\', r'\hline',
                      r'\textbf{Form} & & ' + HEAD + ' & ' + HEAD + r' \\', r'\hline'] + lines + [r'\end{tabular}']) + '\n'
    open(f'{dest}/sim_long_table.tex', 'w').write(body)
    key['long_min_cover'] = {m: round(float(ML.xs(m, level='method').cover.min()), 3) for m in METHODS}
    for mm in ('BNN-RE-T', 'BNN-RE-C'):
        key[f'long_{mm}_vs_RE_pct_by_block'] = {f'{f}|{k}': round(100 * (ML.loc[(f, mm, 'cons', k), 'width'] / ML.loc[(f, 'BNN-RE', 'cons', k), 'width'] - 1), 1) for (f, m, s, k) in ML.index if m == mm and s == 'cons' and k >= 2}
    key['long_RE_vs_BNN_pct_by_block'] = {f'{f}|{k}': round(100 * (ML.loc[(f, 'BNN-RE', 'cons', k), 'width'] / ML.loc[(f, 'BNN', 'cons', k), 'width'] - 1), 1) for (f, m, s, k) in ML.index if m == 'BNN-RE' and s == 'cons'}
    key['long_cover_by_block'] = {mm: {f'{f}|{k}': round(float(ML.loc[(f, mm, 'cons', k), 'cover']), 3) for (f, m, s, k) in ML.index if m == mm and s == 'cons'} for mm in METHODS}
    for f, _ in FORMS:
        if f in SL.form.unique():
            M = ML.xs(f, level='form')
            for m in METHODS:
                key[f'long|{f}|{m}|cons|width_upd20'] = round(float(M.loc[(m, 'cons', 21), 'width']), 3); key[f'long|{f}|{m}|cons|cover_upd20'] = round(float(M.loc[(m, 'cons', 21), 'cover']), 3)

key['alpha_cal'] = ALPHA_CAL
json.dump(key, open(f'{dest}/key_numbers.json', 'w'), indent=1)

# ------------------------------------------------------------------ figures
COL = {'BNN': '#2a78d6', 'BNN-RE': '#eb6834', 'BNN-RE-T': '#1baf7a', 'BNN-RE-C': '#eda100'}
HATCH = {'BNN': None, 'BNN-RE': None, 'BNN-RE-T': '////', 'BNN-RE-C': '....'}
LAB = {'BNN': 'BNN', 'BNN-RE': 'BNN-RE', 'BNN-RE-T': 'BNN-RE-T (tempered variance prior)', 'BNN-RE-C': 'BNN-RE-C (calibrated width)'}
TEXT, GRID = '#0b0b0b', '#d9d8d4'
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#52514e', 'axes.labelcolor': TEXT, 'xtick.color': TEXT, 'ytick.color': TEXT})

def panel(ax, df, col, ylabel, nominal=None):
    slots = [('cons', 1, 'Initial'), ('cons', 2, 'Update #1')] + [(s, u + 1, f'Update #{u}') for u in (2, 3, 4) for s in ('cons', 'cumu')]
    w, pos, ticks, labs, bands = 0.17, 0, [], [], []
    for j, (s, k, lab) in enumerate(slots):
        if j >= 2: bands.append((pos - 0.5, pos + 0.5, s))
        for i, m in enumerate(METHODS):
            v = df[(df.method == m) & (df.strategy == s) & (df.eval_block == k)][col].values
            if len(v) == 0: continue
            bp = ax.boxplot(v, positions=[pos + (i - 1.5) * (w + 0.03)], widths=w, patch_artist=True, showmeans=True,
                            meanprops=dict(marker='^', markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=3.5),
                            medianprops=dict(color=TEXT, linewidth=0.9), whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                            flierprops=dict(marker='o', markersize=2, markerfacecolor='none', markeredgecolor='#52514e'))
            for p in bp['boxes']:
                p.set_facecolor(COL[m]); p.set_edgecolor('#52514e'); p.set_linewidth(0.8)
                if HATCH[m]: p.set_hatch(HATCH[m])
        if j < 2 or s == 'cons': ticks.append(pos if j < 2 else pos + 0.5); labs.append(lab)
        pos += 1
    for a, b, s in bands: ax.axvspan(a, b, color='#f1f0ed' if s == 'cons' else '#dddcd7', zorder=0, lw=0)
    if nominal is not None: ax.axhline(nominal, color='#e34948', ls='--', lw=1.1, zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(labs); ax.set_xlim(-0.6, pos - 0.4); ax.set_ylabel(ylabel); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top',): ax.spines[sp].set_visible(False)
    ax.spines['right'].set_visible(False)

def figure(df, path, tiff, orig_label='Ratio of upper to lower limit\n(original scale)'):
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 5.4))
    panel(axes[0], df, 'width', 'Interval width (log scale)'); axes[0].set_title('95% interval widths', fontsize=9.5)
    # original-scale reading on the left: exp(width) = ratio of the upper to the lower interval limit; log-scale width on the right
    axes[0].yaxis.tick_right(); axes[0].yaxis.set_label_position('right'); axes[0].spines['right'].set_visible(True); axes[0].spines['left'].set_visible(False)
    sec = axes[0].secondary_yaxis('left', functions=(np.exp, np.log))
    lo_, hi_ = np.exp(axes[0].get_ylim()); ticks = [t for t in (6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50) if lo_ <= t <= hi_]
    sec.set_yticks(ticks); sec.set_ylabel(orig_label, fontsize=8.5); sec.tick_params(labelsize=7.5)
    for sp in ('top',): sec.spines[sp].set_visible(False)
    panel(axes[1], df, 'cover', 'Interval coverage rate', nominal=0.95); axes[1].set_title('95% interval coverage rates', fontsize=9.5)
    h = [Patch(facecolor='#f1f0ed', label='Consecutive update'), Patch(facecolor='#dddcd7', label='Cumulative update')]
    h += [Patch(facecolor=COL[m], edgecolor='#52514e', hatch=HATCH[m], label=LAB[m]) for m in METHODS]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)
    from PIL import Image
    Image.open(path).convert('RGB').save(tiff, format='TIFF', compression='tiff_lzw', dpi=(300, 300))

figure(F, f'{FIGDIR}/Figure_2.png', f'{tiffdir}/Figure_2.tif', orig_label='Ratio of upper to lower limit\n(mortality)')
HAS_WEEKLY = len(sys.argv) > 7 and bool(glob.glob(f'{sys.argv[7]}/perobs_seed*.npz'))
if S5 is not None:
    for n, (f, _) in enumerate(FORMS, start=4 if HAS_WEEKLY else 3):
        if f in S5.form.unique(): figure(S5[S5.form == f], f'{FIGDIR}/Figure_{n}.png', f'{tiffdir}/Figure_{n}.tif')
print(json.dumps(key, indent=1)[:3000])

# ------------------------------------------------------------------ weekly updating of the field data (optional 7th argument)
if len(sys.argv) > 7 and glob.glob(f'{sys.argv[7]}/perobs_seed*.npz'):
    out_w = sys.argv[7]
    blocks_info = pd.read_csv(f'{out_w}/blocks.csv'); KW = len(blocks_info)
    rows = []
    for f in sorted(glob.glob(f'{out_w}/perobs_seed*.npz')):
        seed = int(f.split('seed')[1].split('.')[0]); z = np.load(f)
        for strat in ('cons', 'cumu'):
            seqs = {m: [z[f'{m}|initial|1']] + [z[f'{m}|{strat}|{k}'] for k in range(2, KW + 1)] for m in ('BNN', 'BNN-RE', 'BNN-RE-T')}
            for m, bl in seqs.items():
                for k, arr in enumerate(bl):
                    lo, hi, _, y = arr
                    rows.append(dict(seed=seed, method=m, strategy=strat, eval_block=k + 1, width=float(np.mean(hi - lo)), cover=float(np.mean((lo < y) & (y < hi))), n=len(y)))
            for k, r in enumerate(calibrate_sequence(seqs['BNN-RE'], alpha=ALPHA_CAL)):
                rows.append(dict(seed=seed, method='BNN-RE-C', strategy=strat, eval_block=k + 1, width=r['width'], cover=r['cover'], n=len(seqs['BNN-RE'][k][3])))
    W = pd.DataFrame(rows); key['weekly_seeds'] = int(W.seed.nunique()); key['weekly_blocks'] = KW
    MW = W.groupby(['method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    # pooled coverage over all evaluated groups (blocks weighted by size) and mean width, per method and schedule, over the updates
    Wu = W[W.eval_block >= 2].copy(); Wu['hits'] = Wu.cover * Wu.n
    pooled = Wu.groupby(['method', 'strategy']).apply(lambda d: pd.Series(dict(width=d.width.mean(), cover=d.hits.sum() / d.n.sum())))
    key['weekly_pooled'] = {f'{m}|{s}': dict(width=round(float(v.width), 3), cover=round(float(v.cover), 3)) for (m, s), v in pooled.iterrows()}
    key['weekly_min_block_cover'] = {m: round(float(MW.xs(m, level='method').cover.min()), 3) for m in METHODS}
    # table: windows of five updates (widths averaged over blocks and seeds; coverage pooled over all evaluated groups in the window)
    WINDOWS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, KW - 1)]
    def win_stats(df, s, lo, hi):
        d = df[(df.strategy == s) & (df.eval_block >= lo + 1) & (df.eval_block <= hi + 1)].copy(); d['hits'] = d.cover * d.n
        return d.groupby('method').apply(lambda g: pd.Series(dict(width=g.width.mean(), cover=g.hits.sum() / g.n.sum())))
    init = W[W.eval_block == 1].groupby('method').apply(lambda g: pd.Series(dict(width=g.width.mean(), cover=(g.cover * g.n).sum() / g.n.sum())))
    lines = [r'\multicolumn{2}{l|}{Initial model (week 1)} & ' + ' & '.join('%.2f' % init.loc[m, 'width'] for m in METHODS) + ' & ' + ' & '.join('%.3f' % init.loc[m, 'cover'] for m in METHODS) + r' \\', r'\hline']
    for lo, hi in WINDOWS:
        wk_lo = blocks_info.weeks.iloc[lo]; wk_hi = blocks_info.weeks.iloc[hi]
        wk_hi = wk_hi.split(',')[-1]
        for j, s_ in enumerate(('cons', 'cumu')):
            st = win_stats(W, s_, lo, hi)
            head = (r'\multirow{2}{*}{Updates \#%d--\#%d (weeks %s--%s)}' % (lo, hi, wk_lo, wk_hi)) if j == 0 else ''
            lines.append(head + ' & ' + ('Cons.' if s_ == 'cons' else 'Cumu.') + ' & ' + ' & '.join('%.2f' % st.loc[m, 'width'] for m in METHODS) + ' & ' + ' & '.join('%.3f' % st.loc[m, 'cover'] for m in METHODS) + r' \\')
        lines.append(r'\hline')
    for j, s_ in enumerate(('cons', 'cumu')):
        head = r'\multirow{2}{*}{All updates \#1--\#%d}' % (KW - 1) if j == 0 else ''
        lines.append(head + ' & ' + ('Cons.' if s_ == 'cons' else 'Cumu.') + ' & ' + ' & '.join('%.2f' % pooled.loc[(m, s_), 'width'] for m in METHODS) + ' & ' + ' & '.join('%.3f' % pooled.loc[(m, s_), 'cover'] for m in METHODS) + r' \\')
    body = '\n'.join([r'\begin{tabular}{lc|cccc|cccc}', r'\hline', r'&& \multicolumn{4}{c|}{\textbf{Average Interval Width}} & \multicolumn{4}{c}{\textbf{Interval Coverage Rate}} \\', r'\hline',
                      r'&& ' + HEAD + ' & ' + HEAD + r' \\', r'\hline'] + lines + [r'\hline', r'\end{tabular}']) + '\n'
    key['weekly_windows'] = {f'{lo}-{hi}|{s_}': {m: dict(width=round(float(win_stats(W, s_, lo, hi).loc[m, 'width']), 3), cover=round(float(win_stats(W, s_, lo, hi).loc[m, 'cover']), 3)) for m in METHODS} for lo, hi in WINDOWS for s_ in ('cons', 'cumu')}
    key['weekly_by_block'] = {f'{m}|{s_}': [round(float(MW.loc[(m, s_, k), 'width']), 3) for k in range(1, KW + 1) if (m, s_, k) in MW.index] for m in METHODS for s_ in ('cons', 'cumu')}
    open(f'{dest}/weekly_table.tex', 'w').write(body)
    json.dump(key, open(f'{dest}/key_numbers.json', 'w'), indent=1)
    # trajectory figure: mean width (ratio on the left, log width on the right) and coverage pooled over seeds per block, both schedules
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 5.2), sharex=True)
    xs = np.arange(1, KW + 1)
    for m in METHODS:
        for s_, ls in (('cumu', '-'), ('cons', '--')):
            mw = MW.xs((m, s_), level=('method', 'strategy')).reindex(xs)
            if s_ == 'cons': mw.iloc[0] = MW.xs((m, 'cumu'), level=('method', 'strategy')).reindex(xs).iloc[0]
            axes[0].plot(xs, mw.width.values, color=COL[m], lw=1.5, ls=ls, marker='o' if s_ == 'cumu' else None, ms=2.6)
            axes[1].plot(xs, mw.cover.values, color=COL[m], lw=1.5, ls=ls, marker='o' if s_ == 'cumu' else None, ms=2.6)
    for ax in axes:
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines['top'].set_visible(False)
    axes[0].set_ylabel('Interval width (log scale)'); axes[0].yaxis.tick_right(); axes[0].yaxis.set_label_position('right'); axes[0].spines['left'].set_visible(False)
    sec = axes[0].secondary_yaxis('left', functions=(np.exp, np.log)); lo_, hi_ = np.exp(axes[0].get_ylim())
    sec.set_yticks([t for t in (8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22) if lo_ <= t <= hi_]); sec.set_ylabel('Ratio of upper to lower limit\n(mortality)', fontsize=8.5); sec.tick_params(labelsize=7.5)
    axes[0].set_title('95% interval widths, weekly updating', fontsize=9.5)
    axes[1].axhline(0.95, color='#e34948', ls='--', lw=1.1); axes[1].set_ylabel('Interval coverage rate'); axes[1].spines['right'].set_visible(False)
    axes[1].set_title('95% interval coverage rates', fontsize=9.5); axes[1].set_xlabel('Evaluated weekly block (block 1: initial model; block k: forecast after Update #k-1)')
    axes[1].set_xticks(xs[::2]); axes[1].set_xticklabels([str(k) for k in xs[::2]])
    from matplotlib.lines import Line2D
    h = [Patch(facecolor=COL[m], edgecolor='#52514e', hatch=HATCH[m], label=LAB[m]) for m in METHODS]
    h += [Line2D([0], [0], color=TEXT, lw=1.5, ls='-', marker='o', ms=2.6, label='Cumulative update'), Line2D([0], [0], color=TEXT, lw=1.5, ls='--', label='Consecutive update')]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0)); fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f'{FIGDIR}/Figure_2.png', dpi=300); plt.close(fig)
    from PIL import Image
    Image.open(f'{FIGDIR}/Figure_2.png').convert('RGB').save(f'{tiffdir}/Figure_2.tif', format='TIFF', compression='tiff_lzw', dpi=(300, 300))
    print('weekly:', json.dumps({k_: v for k_, v in key.items() if k_.startswith('weekly')}, indent=1))
