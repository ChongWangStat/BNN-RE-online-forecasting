"""Data preparation for the swine nursery-mortality study.

Replicates the preprocessing of the original notebooks (cons_re - RUN.ipynb, cells 3-26)
step for step, and returns tensors for the training set and the five 2023 test sets.
Only the bookkeeping differs: every design matrix Z is built against the full list of
training sow sources, by name, so that source effects are matched by identity.
"""
import numpy as np, pandas as pd, torch

def load(xlsx_path):
    # pandas' default NA list contains the string "None"; in this workbook "None" is a
    # medication category, so it is kept as text and only genuinely blank cells are missing
    na = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
          '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
    dat = pd.read_excel(xlsx_path, keep_default_na=False, na_values=na)
    n_raw = dat.shape[0]
    dat['SowFarmMed'] = dat['SowFarmMed'].replace("None", "none")
    dat['WeanPigShot'] = dat['WeanPigShot'].replace("None", "none")
    dat['SowFarmMed'] = dat['SowFarmMed'].replace("none", "no")
    dat['WeanPigShot'] = dat['WeanPigShot'].replace("none", "no")
    dat = dat.dropna().sort_values(by=['year_num', 'week_num', 'CloseDate', 'SowSource'])
    dat.drop(columns=['lot_no2', 'lot_nur', 'CloseDate', 'LotPrefix', 'HeadIn', 'mortality_60days', 'ADG'], inplace=True)
    dat = dat.drop_duplicates()
    dat = dat[dat['avg_mortality_60days'] > 0].copy()
    dat['avg_mortality_60days'] = np.log(dat['avg_mortality_60days'])
    num_dat = dat.select_dtypes(include=np.number)
    cat_dat = dat.select_dtypes(include=['object', 'string'])
    cat_cols = [c for c in cat_dat.columns if c != 'SowSource']
    cat_dat = pd.get_dummies(cat_dat, columns=cat_cols)
    combined = pd.concat([num_dat, cat_dat], axis=1)
    combined = combined[[c for c in combined if c != 'avg_mortality_60days'] + ['avg_mortality_60days']]
    test = combined[combined['year_num'] == 2023].copy()
    train = combined[combined['year_num'] != 2023].copy()
    train.drop(columns=['year_num'], inplace=True); test.drop(columns=['year_num'], inplace=True)
    # keep 2023 groups whose sow source appears in training
    seen = set(train['SowSource'].unique())
    n_test_all = test.shape[0]
    test = test[test['SowSource'].isin(seen)].copy()
    blocks = [test[test['week_num'] <= 5],
              test[(test['week_num'] > 5) & (test['week_num'] <= 10)],
              test[(test['week_num'] > 10) & (test['week_num'] <= 15)],
              test[(test['week_num'] > 15) & (test['week_num'] <= 20)],
              test[test['week_num'] > 20]]
    info = dict(n_raw=n_raw, n_clean=combined.shape[0], n_train=train.shape[0], n_test_all=n_test_all,
                n_test=test.shape[0], n_sources_train=train['SowSource'].nunique(),
                n_sources_test=test['SowSource'].nunique(), block_sizes=[b.shape[0] for b in blocks],
                n_numeric=len([c for c in num_dat.columns if c not in ('year_num', 'avg_mortality_60days')]),
                n_categorical=len(cat_cols), n_inputs=combined.shape[1] - 3)  # minus SowSource, response, year
    return train, blocks, info

class Design:
    """Holds the training-range scaling and the full source list; builds (x, y, z) by name."""
    def __init__(self, train):
        self.levels = sorted(train['SowSource'].unique())
        self.idx = {s: j for j, s in enumerate(self.levels)}
        xcols = [c for c in train.columns if c not in ('SowSource', 'avg_mortality_60days')]
        self.xcols = xcols
        X = torch.tensor(train[xcols].values.astype(np.float32))
        self.min_x = X.min(dim=0).values
        rng = X.max(dim=0).values - self.min_x
        rng[rng == 0] = 1.0
        self.range_x = rng
    def xyz(self, df):
        X = torch.tensor(df[self.xcols].values.astype(np.float32))
        X = (X - self.min_x) / self.range_x
        y = torch.tensor(df['avg_mortality_60days'].values.astype(np.float32))
        Z = torch.zeros(len(df), len(self.levels))
        for i, s in enumerate(df['SowSource'].tolist()):
            Z[i, self.idx[s]] = 1.0
        return X, y, Z, df['SowSource'].tolist()

if __name__ == '__main__':
    import sys, json
    train, blocks, info = load(sys.argv[1])
    print(json.dumps(info, indent=1))
    d = Design(train)
    x, y, z, _ = d.xyz(train)
    print('train x', tuple(x.shape), 'z', tuple(z.shape), 'y sd', float(y.std()))
    for k, b in enumerate(blocks):
        xb, yb, zb, _ = d.xyz(b); print('block', k + 1, tuple(xb.shape), 'sources', b['SowSource'].nunique(), 'y sd', round(float(yb.std()), 3))
