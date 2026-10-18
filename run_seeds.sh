#!/bin/bash
# one Python process per seed (the JIT ELBO caches compiled graphs, so a long-lived process grows in memory)
OUT=$1; shift
for s in "$@"; do
  [ -f $OUT/summary_seed$s.csv ] && continue
  python3 -W ignore run_real.py data.xlsx $OUT $s
done
