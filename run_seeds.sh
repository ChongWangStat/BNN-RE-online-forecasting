#!/bin/bash
# Field study, BNN / BNN-RE / LMM-RE: one Python process per seed (the JIT ELBO caches compiled graphs, so a long-lived
# process grows in memory).  Usage: ./run_seeds.sh OUTDIR SEED [SEED ...]   (data.xlsx in this directory)
cd "$(dirname "$0")"
OUT=$1; shift
for s in "$@"; do
  [ -f $OUT/summary_seed$s.csv ] && continue
  python3 -W ignore run_real.py data.xlsx $OUT $s
done
