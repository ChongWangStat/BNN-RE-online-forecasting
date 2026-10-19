#!/bin/bash
# Field study, BNN-RE-T: one process per seed.  Usage: ./run_temper_seeds.sh OUTDIR N0 SEED [SEED ...]
cd "$(dirname "$0")"
OUT=$1; N0=$2; shift 2
for s in "$@"; do [ -f $OUT/summary_seed$s.csv ] && continue; python3 -W ignore run_real_tempered.py data.xlsx $OUT $N0 $s; done
