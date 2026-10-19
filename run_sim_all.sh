#!/bin/bash
# Simulation study driver: all three functional forms for (i) the 5-block design with BNN, BNN-RE, BNN-RE-T and LMM-RE,
# (ii) the reverse-drift scenario (residual SD rising from 0.62 to 0.75 instead of falling to 0.50), and
# (iii) the 20-update horizon (21 blocks of 50, consecutive schedule).  Usage: ./run_sim_all.sh data.xlsx [n_reps]
# Outputs: out_sim/ (sim_<form>_rep*.csv, perobs_<form>.npz), out_sim_rev/ (sim_<form>_rev_rep*.csv), out_sim_long/ (sim_<form>_K21_rep*.csv).
set -e
cd "$(dirname "$0")"
XLSX=${1:-data.xlsx}; R=${2:-100}
for form in linear nonlinear nn; do
  [ -f out_sim/sim_${form}_rep$((R-1)).csv ] || METHODS=BNN,BNN-RE,LMM-RE VARIANTS=a,c python3 -W ignore sim_batched.py $XLSX out_sim $form $R 5
done
for form in linear nonlinear nn; do
  [ -f out_sim_rev/sim_${form}_rev_rep$((R-1)).csv ] || SE_TE=0.75 TAG=_rev METHODS=BNN,BNN-RE VARIANTS=a,c python3 -W ignore sim_batched.py $XLSX out_sim_rev $form $R 5
done
for form in linear nonlinear nn; do
  [ -f out_sim_long/sim_${form}_K21_rep$((R-1)).csv ] || CUMU=0 METHODS=BNN,BNN-RE VARIANTS=a,c python3 -W ignore sim_batched.py $XLSX out_sim_long $form $R 21
done
echo "simulations done: $(date)"
