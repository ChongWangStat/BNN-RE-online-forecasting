#!/bin/bash
OUT=$1; N0=$2; shift 2
for s in "$@"; do [ -f $OUT/summary_seed$s.csv ] && continue; python3 -W ignore run_real_tempered.py data.xlsx $OUT $N0 $s; done
