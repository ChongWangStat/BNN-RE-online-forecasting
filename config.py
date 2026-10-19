"""Shared settings for the field study (run_real.py, run_real_tempered.py).
HID, LR, STEPS: hidden units, Adam learning rate and epochs selected by five-fold cross-validation on the training
records (cv_hyper.py; results in results_cv/cv_results.csv)."""
HID, LR, STEPS = 10, 0.005, 1500
NDRAW = 5000            # posterior predictive draws per interval
JIT = True              # Pyro's JIT-compiled Trace_ELBO: same objective, about four times faster per fit
N0 = 60                 # tempered prior: the previous posterior counts as one block of groups (run_real_tempered.py argument)
METHODS_REAL = (('BNN', False, HID), ('BNN-RE', True, HID), ('LMM-RE', True, 0))   # (name, random effect, hidden units; 0 = linear mean)
