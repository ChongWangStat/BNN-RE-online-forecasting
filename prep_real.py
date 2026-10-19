"""Data preparation for the production records (revision 2: one record per nursery lot, corrected units, winsorised inputs).

Steps, in order (counts reported by the __main__ block are the ones quoted in the paper):
  1. Read the workbook; "None" is a medication category and is kept as text; only blank cells are missing.
  2. Drop records with any missing field.
  3. Pre-weaning mortality is a proportion in most records but was entered as a percentage in some (values above 1);
     those are divided by 100.
  4. The database holds one record per nursery sub-lot (barn); sub-lots of the same nursery lot share the lot-level
     predictors and the lot's head-weighted average mortality, which is the outcome.  One record is kept per nursery
     lot: that of the largest sub-lot by head placed, with the health-event fields recorded during the nursery phase
     (lateral coronavirus, PRRSV and enteric introductions, other health issues) aggregated over the lot's sub-lots
     (an event in any barn counts for the lot).
  5. Groups with zero mortality would be dropped before the log transform (none occur).
  6. Numerical predictors are winsorised at the 1st and 99th percentiles of the training records (a handful of rolling
     sow-farm indicators contain impossible values, e.g. 1,574 litters per female per year, from near-zero denominators)
     and then rescaled to [0, 1] with the training range (Design); categorical fields are one-hot encoded.
The sow source is used as recorded (a group weaned from more than one farm carries the combined label as its own
source level) and enters only through the random effect.
"""
import numpy as np, pandas as pd, torch

ID_COLS = ['lot_no2', 'lot_nur', 'CloseDate', 'LotPrefix', 'HeadIn', 'mortality_60days', 'ADG']
RESPONSE = 'avg_mortality_60days'
WINSOR = (0.01, 0.99)

def read_raw(xlsx_path):
    na = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
          '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    dat = pd.read_excel(xlsx_path, keep_default_na=False, na_values=na)
    for c in ('SowFarmMed', 'WeanPigShot'):
        dat[c] = dat[c].replace('None', 'none').replace('none', 'no')
    return dat

EVENT_FIELDS = ['LateralCoronavirus', 'LateralPRRS', 'LateralEnteric', 'OtherHealthIssue']   # recorded during the nursery phase, per barn

def _aggregate_events(g):
    """Lot-level value of each nursery-phase event field from its sub-lot (barn) records: an event in any barn counts."""
    out = {}
    for c in EVENT_FIELDS:
        vals = [v for v in g[c].tolist() if v != 'no']
        if not vals: out[c] = 'no'
        elif c == 'LateralEnteric' and {'Ecoli', 'Salmonella'} <= set(v for x in vals for v in x.split('_')): out[c] = 'Ecoli_Salmonella'
        else: out[c] = pd.Series(vals).mode().iloc[0]
    return out

def clean(xlsx_path):
    """Returns the cleaned per-lot table (unscaled, categorical as text) and the cleaning counts."""
    dat = read_raw(xlsx_path); info = dict(n_raw=dat.shape[0])
    dat = dat.dropna(); info['n_complete'] = dat.shape[0]
    pw = dat['Pre_weaning_mortality'].astype(float)
    info['n_pwm_percent'] = int((pw > 1).sum())
    dat['Pre_weaning_mortality'] = np.where(pw > 1, pw / 100.0, pw)
    info['n_lots'] = int(dat['lot_nur'].nunique())
    info['n_lots_multi_sublot'] = int((dat.groupby('lot_nur').size() > 1).sum())
    agg = dat.groupby('lot_nur').apply(lambda g: pd.Series(_aggregate_events(g)))
    dat = dat.sort_values(['lot_nur', 'HeadIn'], ascending=[True, False]).drop_duplicates(subset='lot_nur', keep='first')
    for c in EVENT_FIELDS:
        dat[c] = dat['lot_nur'].map(agg[c])
    info['n_lots_event_in_other_barn'] = int(sum((dat.set_index('lot_nur')[c] != 'no').sum() for c in EVENT_FIELDS))
    dat = dat.sort_values(by=['year_num', 'week_num', 'CloseDate', 'SowSource'])
    dat = dat.drop(columns=ID_COLS)
    info['n_zero_mortality'] = int((dat[RESPONSE] <= 0).sum())
    dat = dat[dat[RESPONSE] > 0].copy()
    dat[RESPONSE] = np.log(dat[RESPONSE].astype(float))
    info['n_clean'] = dat.shape[0]
    return dat.reset_index(drop=True), info

def load(xlsx_path):
    dat, info = clean(xlsx_path)
    num_cols = [c for c in dat.select_dtypes(include=np.number).columns if c not in ('year_num', RESPONSE)]
    cat_cols = [c for c in dat.select_dtypes(include=['object', 'string']).columns if c != 'SowSource']
    # winsorise numerical predictors at the training percentiles, then one-hot encode
    train_mask = dat['year_num'] != 2023
    caps = {c: (dat.loc[train_mask, c].quantile(WINSOR[0]), dat.loc[train_mask, c].quantile(WINSOR[1])) for c in num_cols}
    for c, (lo, hi) in caps.items():
        dat[c] = dat[c].astype(float).clip(lo, hi)
    cat_dat = pd.get_dummies(dat[['SowSource'] + cat_cols], columns=cat_cols).astype({c: float for c in []})
    combined = pd.concat([dat[['year_num'] + num_cols], cat_dat, dat[[RESPONSE]]], axis=1)
    for c in combined.columns:
        if combined[c].dtype == bool: combined[c] = combined[c].astype(float)
    test = combined[combined['year_num'] == 2023].copy()
    train = combined[combined['year_num'] != 2023].copy()
    train.drop(columns=['year_num'], inplace=True); test.drop(columns=['year_num'], inplace=True)
    seen = set(train['SowSource'].unique())
    n_test_all = test.shape[0]
    test = test[test['SowSource'].isin(seen)].copy()
    blocks = [test[test['week_num'] <= 5],
              test[(test['week_num'] > 5) & (test['week_num'] <= 10)],
              test[(test['week_num'] > 10) & (test['week_num'] <= 15)],
              test[(test['week_num'] > 15) & (test['week_num'] <= 20)],
              test[test['week_num'] > 20]]
    info.update(n_train=train.shape[0], n_test_all=n_test_all, n_test=test.shape[0],
                n_sources_train=train['SowSource'].nunique(), n_sources_test_all=int(combined[combined['year_num'] == 2023]['SowSource'].nunique()),
                n_sources_test=test['SowSource'].nunique(), block_sizes=[b.shape[0] for b in blocks],
                n_numeric=len(num_cols), n_categorical=len(cat_cols), n_inputs=combined.shape[1] - 3,
                winsor_caps={c: (float(lo), float(hi)) for c, (lo, hi) in caps.items()})
    return train, blocks, info

class Design:
    """Holds the training-range scaling and the full source list; builds (x, y, z) by name."""
    def __init__(self, train):
        self.levels = sorted(train['SowSource'].unique())
        self.idx = {s: j for j, s in enumerate(self.levels)}
        xcols = [c for c in train.columns if c not in ('SowSource', RESPONSE)]
        self.xcols = xcols
        X = torch.tensor(train[xcols].values.astype(np.float32))
        self.min_x = X.min(dim=0).values
        rng = X.max(dim=0).values - self.min_x
        rng[rng == 0] = 1.0
        self.range_x = rng
    def xyz(self, df):
        X = torch.tensor(df[self.xcols].values.astype(np.float32))
        X = (X - self.min_x) / self.range_x
        y = torch.tensor(df[RESPONSE].values.astype(np.float32))
        Z = torch.zeros(len(df), len(self.levels))
        for i, s in enumerate(df['SowSource'].tolist()):
            Z[i, self.idx[s]] = 1.0
        return X, y, Z, df['SowSource'].tolist()

if __name__ == '__main__':
    import sys, json
    train, blocks, info = load(sys.argv[1])
    print(json.dumps({k: v for k, v in info.items() if k != 'winsor_caps'}, indent=1))
    d = Design(train)
    x, y, z, _ = d.xyz(train)
    print('train x', tuple(x.shape), 'z', tuple(z.shape), 'y sd', round(float(y.std()), 3))
    for k, b in enumerate(blocks):
        xb, yb, zb, _ = d.xyz(b); print('block', k + 1, tuple(xb.shape), 'sources', b['SowSource'].nunique(), 'y sd', round(float(yb.std()), 3))
