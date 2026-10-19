#!/bin/bash
# Field study driver: 50 seeds for BNN / BNN-RE / LMM-RE, then BNN-RE-T (n0 = 60).  Usage: ./run_field_all.sh
cd "$(dirname "$0")"
./run_seeds.sh out_real $(seq 567 616)
./run_temper_seeds.sh out_real_t 60 $(seq 567 616)
./run_sensitivity.sh
echo "field runs done: $(date)"
