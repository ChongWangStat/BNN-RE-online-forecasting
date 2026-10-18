"""Conformal-style calibration of the Bayesian prediction intervals, computed after the fact from the saved
per-observation predictions (each block was predicted before its outcomes were seen).

For the prediction of block k+1 (made after Update #k), the calibration set is the set of test observations
in blocks 1..k together with the intervals that were issued for them at the time.  Each calibration
observation gets the standardised score s_i = |y_i - c_i| / h_i, where c_i and h_i are the centre and the
half-width of its 95% Bayesian interval.  With n calibration scores, q is the ceil((n+1)(1-alpha))/n
empirical quantile of the scores (split-conformal correction), and the calibrated interval for every
observation in block k+1 is c +/- q h.  If the Bayesian intervals over-cover on the calibration blocks, q < 1
and the intervals shrink; if they under-cover, q > 1 and the intervals widen.  The initial model's
prediction of block 1 has no calibration set and is reported as the Bayesian interval, as is any block whose
calibration set holds fewer than min_n = 50 scores (with few scores the quantile is the largest score and the
calibrated interval is erratic; 50 is about one five-week block of groups in the production data).
"""
import numpy as np

def conformal_q(scores, alpha=0.05):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n: return float(np.inf)
    return float(np.sort(scores)[k - 1])

def calibrate_sequence(blocks, alpha=0.05, window=None, min_n=50):
    """blocks: list over eval blocks 1..K of arrays [4, n] = (lo, hi, mean, y) for the interval issued for that block.
    Returns per block: (width, coverage, q) of the calibrated intervals; block 1 keeps the Bayesian interval (q = 1)."""
    out = []; score_hist = []
    for k, arr in enumerate(blocks):
        lo, hi, _, y = arr
        c, h = (lo + hi) / 2, (hi - lo) / 2
        if k == 0 or len(score_hist) == 0:
            q = 1.0
        else:
            pool = np.concatenate(score_hist[-window:] if window else score_hist)
            q = conformal_q(pool, alpha) if len(pool) >= min_n else 1.0   # fewer than min_n scores: keep the Bayesian interval
            if not np.isfinite(q): q = 1.0
        lo_c, hi_c = c - q * h, c + q * h
        out.append(dict(width=float(np.mean(hi_c - lo_c)), cover=float(np.mean((lo_c < y) & (y < hi_c))), q=q))
        score_hist.append(np.abs(y - c) / h)
    return out
