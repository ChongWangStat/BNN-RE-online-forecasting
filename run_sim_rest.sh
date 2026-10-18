#!/bin/bash
# finish the 5-block simulations, then the 20-update (21-block) long horizon with the consecutive schedule only
cd /home/claude/bnnre
export VARIANTS=a,c
while pgrep -f "out_sim2" > /dev/null; do sleep 20; done
[ -f out_sim2/perobs_linear.npz ] || python3 -W ignore sim_batched.py data.xlsx out_sim2 linear 100 5
export CUMU=0
for f in nn nonlinear linear; do [ -f out_sim_long/perobs_${f}_K21.npz ] || python3 -W ignore sim_batched.py data.xlsx out_sim_long $f 100 21; done
echo ALLDONE
