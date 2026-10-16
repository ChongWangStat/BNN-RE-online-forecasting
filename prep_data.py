"""Shared data preparation replicating the student's notebooks exactly."""
import numpy as np, pandas as pd, torch
import os
DATA = os.environ.get("BNNRE_DATA", "data/COMPLETE_DATA_MAR_2024.xlsx")  # proprietary production records; not distributed

def load_raw():
    dat = pd.read_excel(DATA)
    dat['SowFarmMed'] = dat['SowFarmMed'].replace("None", "none").fillna('none').replace('none', 'no')
    dat['WeanPigShot'] = dat['WeanPigShot'].replace("None", "none").fillna('none').replace('none', 'no')
    dat = dat.dropna().sort_values(by=['year_num', 'week_num', 'CloseDate', 'SowSource'])
    dat = dat.drop(columns=['lot_no2', 'lot_nur', 'CloseDate', 'LotPrefix', 'HeadIn', 'mortality_60days', 'ADG'])
    dat = dat.drop_duplicates()
    dat = dat[dat['avg_mortality_60days'] > 0].copy()
    dat['avg_mortality_60days'] = np.log(dat['avg_mortality_60days'])
    return dat

def design(dat):
    """One-hot encode categoricals (except SowSource); response last; keep year/week for splitting."""
    num = dat.select_dtypes(include=np.number)
    cat = dat.select_dtypes(include=['object', 'str'])
    cat = pd.get_dummies(cat, columns=[c for c in cat.columns if c != 'SowSource'], dtype=float)
    comb = pd.concat([num, cat], axis=1)
    comb = comb[[c for c in comb if c != 'avg_mortality_60days'] + ['avg_mortality_60days']]
    return comb

class Scaler:
    def __init__(self, X):
        self.min = X.min(0).values; rng = X.max(0).values - self.min; rng[rng == 0] = 1; self.rng = rng
    def __call__(self, X):
        return (X - self.min) / self.rng

def xy(df, pred_cols):
    X = torch.tensor(df[pred_cols].values.astype(np.float32))
    y = torch.tensor(df['avg_mortality_60days'].values.astype(np.float32))
    return X, y

def zmat(sources, levels):
    """N x len(levels) indicator matrix aligned to `levels` (list of source names)."""
    idx = {s: j for j, s in enumerate(levels)}
    Z = torch.zeros(len(sources), len(levels))
    for i, s in enumerate(sources):
        if s in idx: Z[i, idx[s]] = 1.0
    return Z

def real_data_split():
    dat = load_raw(); comb = design(dat)
    test = comb[comb.year_num == 2023]; train = comb[comb.year_num != 2023]
    seen = set(train.SowSource)
    test = test[test.SowSource.isin(seen)]           # as in the notebooks: only sources seen in training
    pred_cols = [c for c in comb.columns if c not in ('SowSource', 'avg_mortality_60days', 'year_num')]
    return train, test, pred_cols
