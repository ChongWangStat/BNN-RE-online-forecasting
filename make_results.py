"""Build all tables, figures and key numbers for the manuscript from the saved run outputs.

Methods reported: BNN; BNN-RE (carry-forward update); BNN-RE-T (tempered sigma_e prior); BNN-RE-C (BNN-RE with
conformal-style calibration of the interval width on the test blocks already observed, computed post hoc from the saved
per-observation predictions); LMM-RE (Bayesian linear mixed model with the same random intercept and the same update).

Usage: python make_results.py OUT_REAL OUT_REAL_T OUT_SIM OUT_SIM_LONG DEST TIFFDIR [OUT_SIM_REV]
When the per-observation prediction files (perobs_*.npz) are present, the calibrated results are computed from them
and written next to the summaries (calib_seed*.csv, calib_<form>.csv); when they are absent (public code release),
the calibrated results are read from those files, so every table and number regenerates from the released summaries."""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from conformal import calibrate_sequence
out_real, out_t, out_sim, out_long, dest, tiffdir = sys.argv[1:7]
out_rev = sys.argv[7] if len(sys.argv) > 7 else None
os.makedirs(dest, exist_ok=True); os.makedirs(tiffdir, exist_ok=True); os.makedirs(f'{dest}/../figures', exist_ok=True)
FIGDIR = os.path.normpath(f'{dest}/../figures')
METHODS = ['BNN', 'BNN-RE', 'BNN-RE-T', 'BNN-RE-C', 'LMM-RE']
FIG_METHODS = ['BNN', 'BNN-RE', 'BNN-RE-T', 'BNN-RE-C']
ALPHA_CAL = float(os.environ.get('ALPHA_CAL', '0.04'))   # calibration level of BNN-RE-C (0.04 = one-point margin over the nominal 0.05)
HEAD = ' & '.join(r'\textbf{%s}' % m for m in METHODS)
NM = len(METHODS)
key = {'alpha_cal': ALPHA_CAL}

def fix_update1(df):
    df = df[~((df.strategy == 'cumu') & (df.eval_block == 2))].copy()
    c = df[(df.strategy == 'cons') & (df.eval_block == 2)].copy(); c['strategy'] = 'cumu'
    return pd.concat([df, c], ignore_index=True)

# ------------------------------------------------------------------ field data: per seed, per method
rows = []
npz = sorted(glob.glob(f'{out_real}/perobs_seed*.npz'))
if npz:
    for f in npz:
        seed = int(f.split('seed')[1].split('.')[0]); z = np.load(f); zt = np.load(f'{out_t}/perobs_seed{seed}.npz')
        cal = []
        for strat in ('cons', 'cumu'):
            seqs = {'BNN': [z['BNN|initial|1']] + [z[f'BNN|{strat}|{k}'] for k in (2, 3, 4, 5)],
                    'BNN-RE': [z['BNN-RE|initial|1']] + [z[f'BNN-RE|{strat}|{k}'] for k in (2, 3, 4, 5)],
                    'LMM-RE': [z['LMM-RE|initial|1']] + [z[f'LMM-RE|{strat}|{k}'] for k in (2, 3, 4, 5)],
                    'BNN-RE-T': [zt['initial|1']] + [zt[f'{strat}|{k}'] for k in (2, 3, 4, 5)]}
            for m, blocks in seqs.items():
                for k, arr in enumerate(blocks):
                    lo, hi, _, y = arr
                    rows.append(dict(seed=seed, method=m, strategy=strat, eval_block=k + 1, width=float(np.mean(hi - lo)), cover=float(np.mean((lo < y) & (y < hi)))))
            for k, r in enumerate(calibrate_sequence(seqs['BNN-RE'], alpha=ALPHA_CAL)):
                cal.append(dict(seed=seed, method='BNN-RE-C', strategy=strat, eval_block=k + 1, width=r['width'], cover=r['cover'], q=r['q']))
        pd.DataFrame(cal).to_csv(f'{out_real}/calib_seed{seed}.csv', index=False); rows.extend(cal)
else:   # summaries only (code release): BNN, BNN-RE, LMM-RE from summary_seed*.csv; BNN-RE-T from the tempered run; BNN-RE-C from calib_seed*.csv
    for f in sorted(glob.glob(f'{out_real}/summary_seed*.csv')):
        seed = int(f.split('seed')[1].split('.')[0])
        d = pd.read_csv(f); d = d[d.method.isin(['BNN', 'BNN-RE', 'LMM-RE'])]
        t = pd.read_csv(f'{out_t}/summary_seed{seed}.csv'); t['method'] = 'BNN-RE-T'
        c = pd.read_csv(f'{out_real}/calib_seed{seed}.csv')
        for dd in (d, t):
            dd = dd.copy(); dd.loc[dd.strategy == 'initial', 'strategy'] = 'cons'
            i0 = dd[dd.eval_block == 1].copy(); i0['strategy'] = 'cumu'
            rows.extend(pd.concat([dd, i0])[['seed', 'method', 'strategy', 'eval_block', 'width', 'cover']].to_dict('records'))
        rows.extend(c.to_dict('records'))
F = pd.DataFrame(rows); key['n_seeds'] = int(F.seed.nunique())
MF = F.groupby(['method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
SF = F.groupby(['method', 'strategy', 'eval_block'])[['width', 'cover']].sem()

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

def wrap(lines, first_col=None):
    first_col = first_col or ('cc|' + 'c' * NM + '|' + 'c' * NM)
    return '\n'.join([r'\begin{tabular}{%s}' % first_col, r'\hline',
                      r'&& \multicolumn{%d}{c|}{\textbf{Average Interval Width}} & \multicolumn{%d}{c}{\textbf{Average Interval Coverage Rate}} \\' % (NM, NM), r'\hline',
                      '&& ' + HEAD + ' & ' + HEAD + r' \\', r'\hline'] + lines + [r'\hline', r'\end{tabular}']) + '\n'
open(f'{dest}/app_res_table.tex', 'w').write(wrap(table_rows(MF, METHODS)))

# key numbers (field)
def mean_upd(M, m, s, col, ks): return float(np.mean([M.loc[(m, s, k), col] for k in ks if (m, s, k) in M.index]))
for m in METHODS:
    for s in ('cons', 'cumu'):
        key[f'field|{m}|{s}|width'] = round(mean_upd(MF, m, s, 'width', (2, 3, 4, 5)), 3); key[f'field|{m}|{s}|cover'] = round(mean_upd(MF, m, s, 'cover', (2, 3, 4, 5)), 3)
key['field_by_cell'] = {f'{m}|{s}|{k}': dict(width=round(float(v.width), 3), cover=round(float(v.cover), 3)) for (m, s, k), v in MF.iterrows()}
key['field_min_cover'] = {m: round(float(MF.xs(m, level='method').cover.min()), 3) for m in METHODS}
key['field_max_se'] = dict(width=round(float(SF.width.max()), 4), cover=round(float(SF.cover.max()), 4))
key['field_RE_narrower_than_BNN_all'] = bool(all(MF.loc[('BNN-RE', s, k), 'width'] < MF.loc[('BNN', s, k), 'width'] for (m, s, k) in MF.index if m == 'BNN-RE'))
for mm in ('BNN-RE-T', 'BNN-RE-C', 'LMM-RE'):
    key[f'field_{mm}_minus_RE_by_block'] = {f'{s}{k}': round(float(MF.loc[(mm, s, k), 'width'] - MF.loc[('BNN-RE', s, k), 'width']), 3) for (m, s, k) in MF.index if m == mm}
    key[f'field_{mm}_vs_RE_pct_by_block'] = {f'{s}{k}': round(100 * (MF.loc[(mm, s, k), 'width'] / MF.loc[('BNN-RE', s, k), 'width'] - 1), 1) for (m, s, k) in MF.index if m == mm and k >= 2}
key['field_RE_minus_BNN_by_block'] = {f'{s}{k}': round(float(MF.loc[('BNN-RE', s, k), 'width'] - MF.loc[('BNN', s, k), 'width']), 3) for (m, s, k) in MF.index if m == 'BNN-RE'}
# seed-level paired differences (same seed, same cell) and their SEs
P = F.pivot_table(index=['seed', 'strategy', 'eval_block'], columns='method', values='width')
for a, b in (('BNN-RE', 'BNN'), ('BNN-RE-T', 'BNN-RE'), ('BNN-RE-C', 'BNN-RE'), ('LMM-RE', 'BNN-RE')):
    dlt = (P[a] - P[b]).groupby(level=['strategy', 'eval_block'])
    key[f'field_paired_{a}_minus_{b}'] = {f'{s}{k}': dict(mean=round(float(v), 3), se=round(float(e), 4)) for (s, k), v, e in zip(dlt.mean().index, dlt.mean().values, dlt.sem().values)}
_sh = F.groupby(['method', 'strategy', 'eval_block']).cover.apply(lambda v: float((v < 0.95).mean())).round(2)
key['field_share_seeds_below_095'] = {f'{m}|{s}': {int(k): float(v) for k, v in _sh.loc[(m, s)].items()} for (m, s) in {(m, s) for (m, s, k) in _sh.index}}

# ------------------------------------------------------------------ simulations (5 blocks), long horizon, reverse drift
def sim_frame(out, forms, kb, tag_extra=''):
    rows = []
    for form in forms:
        tag = ('' if kb == 5 else f'_K{kb}') + tag_extra
        files = sorted(glob.glob(f'{out}/sim_{form}{tag}_rep*.csv'))
        if not files: continue
        d = fix_update1(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        d['method'] = d.method.replace({'BNN-RE(a)': 'BNN-RE', 'BNN-RE(c)': 'BNN-RE-T'})
        d = d[d.method.isin(['BNN', 'BNN-RE', 'BNN-RE-T', 'LMM-RE'])]
        d.loc[d.strategy == 'initial', 'strategy'] = 'cons'          # initial fit is shared by both schedules
        t0 = d[(d.method == 'BNN-RE') & (d.eval_block == 1)].copy(); t0['method'] = 'BNN-RE-T'   # tempering starts at the first update
        d = pd.concat([d, t0], ignore_index=True)
        i0 = d[d.eval_block == 1].copy(); i0['strategy'] = 'cumu'; d = pd.concat([d, i0], ignore_index=True)   # the initial fit is shared by both schedules
        rows.append(d[['form', 'rep', 'method', 'strategy', 'eval_block', 'width', 'cover']])
        npz_f = f'{out}/perobs_{form}{tag}.npz'; cal_f = f'{out}/calib_{form}{tag}.csv'
        if os.path.exists(npz_f):
            z = np.load(npz_f); R = z['BNN|initial|1'].shape[0]; cal = []
            for strat in [st for st in ('cons', 'cumu') if f'BNN-RE(a)|{st}|{kb}' in z]:
                for r in range(R):
                    blocks = [z['BNN-RE|initial|1'][r]] + [z[f'BNN-RE(a)|{strat}|{k}'][r] for k in range(2, kb + 1)]
                    for k, res in enumerate(calibrate_sequence(blocks, alpha=ALPHA_CAL)):
                        cal.append(dict(form=form, rep=r, method='BNN-RE-C', strategy=strat, eval_block=k + 1, width=res['width'], cover=res['cover']))
            cal = pd.DataFrame(cal); cal.to_csv(cal_f, index=False); rows.append(cal)
        elif os.path.exists(cal_f):
            rows.append(pd.read_csv(cal_f))
    return pd.concat(rows, ignore_index=True) if rows else None

FORMS = [('linear', 'Linear'), ('nonlinear', 'Nonlinear'), ('nn', 'NN')]
def present(M): return [m for m in METHODS if m in M.index.get_level_values('method')]
def head(ms): return ' & '.join(r'\textbf{%s}' % m for m in ms)

def sim_table(MS, forms, path, rows_per_form=8):
    ms = present(MS); n = len(ms); lines = []
    for f, name in forms:
        if f not in MS.index.get_level_values('form'): continue
        M = MS.xs(f, level='form')
        lines.append(r'\multirow{%d}{*}{%s} & Initial & ' % (rows_per_form, name) + ' & '.join(cell(M, m, 'cons', 1, 'width', '%.2f') for m in ms) + ' & ' + ' & '.join(cell(M, m, 'cons', 1, 'cover', '%.3f') for m in ms) + r' \\')
        lines.append(r' & Update \#1 & ' + ' & '.join(cell(M, m, 'cons', 2, 'width', '%.2f') for m in ms) + ' & ' + ' & '.join(cell(M, m, 'cons', 2, 'cover', '%.3f') for m in ms) + r' \\')
        for u in (2, 3, 4):
            for s, lab in (('cons', 'Cons.'), ('cumu', 'Cumu.')):
                lines.append(r' & Update \#%d %s & ' % (u, lab) + ' & '.join(cell(M, m, s, u + 1, 'width', '%.2f') for m in ms) + ' & ' + ' & '.join(cell(M, m, s, u + 1, 'cover', '%.3f') for m in ms) + r' \\')
        lines.append(r'\hline')
    body = '\n'.join([r'\begin{tabular}{cl|%s|%s}' % ('c' * n, 'c' * n), r'\hline', r'&& \multicolumn{%d}{c|}{\textbf{Average Interval Width}} & \multicolumn{%d}{c}{\textbf{Average Interval Coverage Rate}} \\' % (n, n), r'\hline',
                      r'\textbf{Form} & & ' + head(ms) + ' & ' + head(ms) + r' \\', r'\hline'] + lines + [r'\end{tabular}']) + '\n'
    open(path, 'w').write(body)

def sim_keys(S, M, prefix):
    key[f'{prefix}_reps'] = S.groupby('form').rep.nunique().to_dict()
    key[f'{prefix}_min_cover'] = {m: round(float(M.xs(m, level='method').cover.min()), 3) for m in METHODS if m in M.index.get_level_values('method')}
    key[f'{prefix}_by_cell'] = {f'{f}|{m}|{s}|{k}': dict(width=round(float(v.width), 3), cover=round(float(v.cover), 3)) for (f, m, s, k), v in M.iterrows()}
    SE = S.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].sem()
    key[f'{prefix}_max_se'] = dict(width=round(float(SE.width.max()), 4), cover=round(float(SE.cover.max()), 4))
    key[f'{prefix}_RE_narrower_all'] = bool(all(M.loc[(f, 'BNN-RE', s, k), 'width'] < M.loc[(f, 'BNN', s, k), 'width'] for (f, m, s, k) in M.index if m == 'BNN-RE'))
    for mm in ('BNN-RE-T', 'BNN-RE-C', 'LMM-RE'):
        if mm not in M.index.get_level_values('method'): continue
        key[f'{prefix}_{mm}_vs_RE_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (M.loc[(f, mm, s, k), 'width'] / M.loc[(f, 'BNN-RE', s, k), 'width'] - 1), 1) for (f, m, s, k) in M.index if m == mm and k >= 2}
        key[f'{prefix}_{mm}_minus_RE_by_block'] = {f'{f}|{s}{k}': round(float(M.loc[(f, mm, s, k), 'width'] - M.loc[(f, 'BNN-RE', s, k), 'width']), 3) for (f, m, s, k) in M.index if m == mm}
        key[f'{prefix}_{mm}_vs_BNN_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (M.loc[(f, mm, s, k), 'width'] / M.loc[(f, 'BNN', s, k), 'width'] - 1), 1) for (f, m, s, k) in M.index if m == mm and k >= 2}
    key[f'{prefix}_RE_vs_BNN_pct_by_block'] = {f'{f}|{s}{k}': round(100 * (M.loc[(f, 'BNN-RE', s, k), 'width'] / M.loc[(f, 'BNN', s, k), 'width'] - 1), 1) for (f, m, s, k) in M.index if m == 'BNN-RE'}
    key[f'{prefix}_RE_minus_BNN_by_block'] = {f'{f}|{s}{k}': round(float(M.loc[(f, 'BNN-RE', s, k), 'width'] - M.loc[(f, 'BNN', s, k), 'width']), 3) for (f, m, s, k) in M.index if m == 'BNN-RE'}
    for f in M.index.get_level_values('form').unique():
        for m in METHODS:
            if (f, m, 'cumu', 2) in M.index or (f, m, 'cons', 2) in M.index:
                s_ = 'cumu' if (f, m, 'cumu', 3) in M.index else 'cons'
                key[f'{prefix}|{f}|{m}|{s_}|width'] = round(mean_upd(M.xs(f, level='form'), m, s_, 'width', (2, 3, 4, 5)), 3)
                key[f'{prefix}|{f}|{m}|{s_}|cover'] = round(mean_upd(M.xs(f, level='form'), m, s_, 'cover', (2, 3, 4, 5)), 3)

S5 = sim_frame(out_sim, [f for f, _ in FORMS], 5)
if S5 is not None:
    MS = S5.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    sim_table(MS, FORMS, f'{dest}/sim_res_table.tex'); sim_keys(S5, MS, 'sim')

SR = sim_frame(out_rev, [f for f, _ in FORMS], 5, tag_extra='_rev') if out_rev else None
if SR is not None:
    MR = SR.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    sim_table(MR, FORMS, f'{dest}/sim_rev_table.tex'); sim_keys(SR, MR, 'rev')
    key['rev_sigma_e'] = {f: {m: [round(float(v), 3) for v in []] for m in []} for f in []}

SL = sim_frame(out_long, [f for f, _ in FORMS], 21)
if SL is not None:
    ML = SL.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].mean()
    REPORT = [(1, 'Initial', 'cons')] + [(u + 1, r'Update \#%d' % u, 'cons') for u in (1, 5, 10, 15, 20)]
    lines = []; msl = present(ML); nl = len(msl)
    for f, name in FORMS:
        if f not in SL.form.unique(): continue
        M = ML.xs(f, level='form')
        for j, (k, lab, s) in enumerate(REPORT):
            hd = (r'\multirow{%d}{*}{%s} & ' % (len(REPORT), name)) if j == 0 else ' & '
            lines.append(hd + lab + ' & ' + ' & '.join(cell(M, m, s, k, 'width', '%.2f') for m in msl) + ' & ' + ' & '.join(cell(M, m, s, k, 'cover', '%.3f') for m in msl) + r' \\')
        lines.append(r'\hline')
    body = '\n'.join([r'\begin{tabular}{cl|%s|%s}' % ('c' * nl, 'c' * nl), r'\hline', r'&& \multicolumn{%d}{c|}{\textbf{Average Interval Width}} & \multicolumn{%d}{c}{\textbf{Average Interval Coverage Rate}} \\' % (nl, nl), r'\hline',
                      r'\textbf{Form} & & ' + head(msl) + ' & ' + head(msl) + r' \\', r'\hline'] + lines + [r'\end{tabular}']) + '\n'
    open(f'{dest}/sim_long_table.tex', 'w').write(body)
    key['long_reps'] = SL.groupby('form').rep.nunique().to_dict()
    key['long_min_cover'] = {m: round(float(ML.xs(m, level='method').cover.min()), 3) for m in present(ML)}
    SEL = SL.groupby(['form', 'method', 'strategy', 'eval_block'])[['width', 'cover']].sem()
    key['long_max_se'] = dict(width=round(float(SEL.width.max()), 4), cover=round(float(SEL.cover.max()), 4))
    for mm in [m for m in ('BNN-RE-T', 'BNN-RE-C', 'LMM-RE') if m in present(ML)]:
        key[f'long_{mm}_vs_RE_pct_by_block'] = {f'{f}|{k}': round(100 * (ML.loc[(f, mm, 'cons', k), 'width'] / ML.loc[(f, 'BNN-RE', 'cons', k), 'width'] - 1), 1) for (f, m, s, k) in ML.index if m == mm and s == 'cons' and k >= 2}
        key[f'long_{mm}_minus_RE_by_block'] = {f'{f}|{k}': round(float(ML.loc[(f, mm, 'cons', k), 'width'] - ML.loc[(f, 'BNN-RE', 'cons', k), 'width']), 3) for (f, m, s, k) in ML.index if m == mm and s == 'cons'}
        key[f'long_{mm}_vs_BNN_pct_by_block'] = {f'{f}|{k}': round(100 * (ML.loc[(f, mm, 'cons', k), 'width'] / ML.loc[(f, 'BNN', 'cons', k), 'width'] - 1), 1) for (f, m, s, k) in ML.index if m == mm and s == 'cons' and k >= 2}
    key['long_RE_vs_BNN_pct_by_block'] = {f'{f}|{k}': round(100 * (ML.loc[(f, 'BNN-RE', 'cons', k), 'width'] / ML.loc[(f, 'BNN', 'cons', k), 'width'] - 1), 1) for (f, m, s, k) in ML.index if m == 'BNN-RE' and s == 'cons'}
    key['long_RE_minus_BNN_by_block'] = {f'{f}|{k}': round(float(ML.loc[(f, 'BNN-RE', 'cons', k), 'width'] - ML.loc[(f, 'BNN', 'cons', k), 'width']), 3) for (f, m, s, k) in ML.index if m == 'BNN-RE' and s == 'cons'}
    key['long_by_cell'] = {f'{f}|{m}|{k}': dict(width=round(float(v.width), 3), cover=round(float(v.cover), 3)) for (f, m, s, k), v in ML.iterrows() if s == 'cons'}
    # sigma_e trajectory of the BNN-RE (posterior mean, averaged over replicates) at Updates 1, 5, 10, 15, 20
    sig = []
    for f, _ in FORMS:
        files = sorted(glob.glob(f'{out_long}/sim_{f}_K21_rep*.csv'))
        if files:
            d = pd.concat([pd.read_csv(x) for x in files]); d = d[(d.method == 'BNN-RE(a)') & (d.strategy == 'cons')]
            key[f'long_sigma_e_RE|{f}'] = {int(k): round(float(v), 3) for k, v in d.groupby('eval_block').sigma_e.mean().items() if k in (2, 6, 11, 16, 21)}

json.dump(key, open(f'{dest}/key_numbers.json', 'w'), indent=1)

# ------------------------------------------------------------------ figures (the four BNN methods; LMM-RE is reported in the tables)
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
        for i, m in enumerate(FIG_METHODS):
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
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

def figure(df, path, tiff, orig_label='Ratio of upper to lower limit\n(original scale)'):
    fig, axes = plt.subplots(2, 1, figsize=(7.09, 5.4))
    panel(axes[0], df, 'width', 'Interval width (log scale)'); axes[0].set_title('95% interval widths', fontsize=9.5)
    # original-scale reading on the left: exp(width) = ratio of the upper to the lower interval limit; log-scale width on the right
    axes[0].yaxis.tick_right(); axes[0].yaxis.set_label_position('right'); axes[0].spines['right'].set_visible(True); axes[0].spines['left'].set_visible(False)
    sec = axes[0].secondary_yaxis('left', functions=(np.exp, np.log))
    lo_, hi_ = np.exp(axes[0].get_ylim()); ticks = [t for t in (6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50) if lo_ <= t <= hi_]
    sec.set_yticks(ticks); sec.set_ylabel(orig_label, fontsize=8.5); sec.tick_params(labelsize=7.5)
    sec.spines['top'].set_visible(False)
    panel(axes[1], df, 'cover', 'Interval coverage rate', nominal=0.95); axes[1].set_title('95% interval coverage rates', fontsize=9.5)
    h = [Patch(facecolor='#f1f0ed', label='Consecutive update'), Patch(facecolor='#dddcd7', label='Cumulative update')]
    h += [Patch(facecolor=COL[m], edgecolor='#52514e', hatch=HATCH[m], label=LAB[m]) for m in FIG_METHODS]
    fig.legend(handles=h, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)
    from PIL import Image
    Image.open(path).convert('RGB').save(tiff, format='TIFF', compression='tiff_lzw', dpi=(300, 300))

figure(F, f'{FIGDIR}/Figure_2.png', f'{tiffdir}/Figure_2.tif', orig_label='Ratio of upper to lower limit\n(mortality)')
if S5 is not None:
    for n, (f, _) in enumerate(FORMS, start=3):
        if f in S5.form.unique(): figure(S5[S5.form == f], f'{FIGDIR}/Figure_{n}.png', f'{tiffdir}/Figure_{n}.tif')
print(json.dumps({k: v for k, v in key.items() if not k.endswith('by_cell')}, indent=1)[:6000])
