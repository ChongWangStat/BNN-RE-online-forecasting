"""Field data: Bayesian BNN / BNN-RE intervals, the tempered-prior variant, and conformal calibration (pooled past blocks)
applied post hoc to BNN-RE and to the tempered variant.  Prints per-block means over seeds and the share of seeds
with block coverage below 0.95.  Usage: python analyze_improvements.py OUT_MAIN OUT_TEMPERED"""
import sys, glob, numpy as np, pandas as pd
from conformal import calibrate_sequence
out_main, out_t = sys.argv[1], sys.argv[2]
rows = []
def add(seed, method, strat, blocks, calib):
    if calib:
        res = calibrate_sequence(blocks)
        for k, r in enumerate(res): rows.append(dict(seed=seed, method=method, strategy=strat, eval_block=k + 1, width=r['width'], cover=r['cover'], q=r['q']))
    else:
        for k, arr in enumerate(blocks):
            lo, hi, _, y = arr; rows.append(dict(seed=seed, method=method, strategy=strat, eval_block=k + 1, width=float(np.mean(hi - lo)), cover=float(np.mean((lo < y) & (y < hi))), q=1.0))
for f in sorted(glob.glob(f'{out_main}/perobs_seed*.npz')):
    seed = int(f.split('seed')[1].split('.')[0]); z = np.load(f)
    ft = f'{out_t}/perobs_seed{seed}.npz'
    zt = np.load(ft) if glob.glob(ft) else None
    for strat in ('cons', 'cumu'):
        bnn = [z['BNN|initial|1']] + [z[f'BNN|{strat}|{k}'] for k in (2, 3, 4, 5)]
        re_ = [z['BNN-RE|initial|1']] + [z[f'BNN-RE(a)|{strat}|{k}'] for k in (2, 3, 4, 5)]
        add(seed, 'BNN', strat, bnn, False); add(seed, 'BNN-RE', strat, re_, False); add(seed, 'BNN-RE-C', strat, re_, True); add(seed, 'BNN-C', strat, bnn, True)
        if zt is not None:
            t_ = [zt['initial|1']] + [zt[f'{strat}|{k}'] for k in (2, 3, 4, 5)]
            add(seed, 'BNN-RE-T', strat, t_, False); add(seed, 'BNN-RE-TC', strat, t_, True)
D = pd.DataFrame(rows)
M = D.groupby(['method', 'strategy', 'eval_block'])[['width', 'cover', 'q']].mean()
below = D.groupby(['method', 'strategy', 'eval_block']).cover.apply(lambda v: (v < 0.95).mean()).rename('share_below_095')
T = M.join(below)
pd.set_option('display.width', 220)
order = ['BNN', 'BNN-C', 'BNN-RE', 'BNN-RE-T', 'BNN-RE-C', 'BNN-RE-TC']
for strat in ('cons', 'cumu'):
    print(f'\n=== {strat} ===')
    tab = T.xs(strat, level='strategy').unstack('eval_block')
    tab = tab.reindex([m for m in order if m in tab.index])
    print(tab['width'].round(3).to_string()); print(tab['cover'].round(3).to_string()); print(tab['share_below_095'].round(2).to_string())
print('\nmean over updates 2-5 (cumu):')
mm = D[(D.strategy == 'cumu') & (D.eval_block >= 2)].groupby('method')[['width', 'cover']].mean().reindex(order)
mm['width_vs_BNNRE_%'] = 100 * (mm.width / mm.loc['BNN-RE', 'width'] - 1); print(mm.round(3).to_string())
D.to_csv('field_improvements_perseed.csv', index=False)
