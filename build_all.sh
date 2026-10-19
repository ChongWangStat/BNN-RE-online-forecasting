#!/bin/bash
# Rebuild Table 1, the results tables, figures and key numbers, the Frontiers manuscript, the LaTeX source zip and the
# code release from the saved run outputs.  Run from anywhere; paths are relative to this script's directory.
# Inputs: out_real (BNN, BNN-RE, LMM-RE field runs), out_real_t (BNN-RE-T field runs), out_sim (4-update simulations),
#         out_sim_long (20-update simulations), out_sim_rev (reverse-drift simulations).  BNN-RE-C is computed post hoc
#         from the saved per-observation predictions (or read from calib_*.csv when those are absent).
set -e
cd "$(dirname "$0")"
[ -f data.xlsx ] && python3 -W ignore make_table1.py data.xlsx frontiers/tables/table_predictors_main.tex > /dev/null
python3 -W ignore make_results.py out_real out_real_t out_sim out_sim_long frontiers/tables frontiers/submission/figures out_sim_rev > frontiers/tables/key_numbers.log
python3 make_schematic.py frontiers/figures/Figure_1.png frontiers/submission/figures/Figure_1.tif > /dev/null
cd frontiers
pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main > build_main.log 2>&1 || true
grep -E "^!" build_main.log || echo "LaTeX: no errors"
grep -i "undefined\|multiply" build_main.log || echo "LaTeX: no undefined references"
rm -f submission/latex_source.zip
zip -q -r submission/latex_source.zip main.tex refs.bib FrontiersinVancouver.cls Frontiers-Vancouver.bst tables figures logo1.eps logo1.pdf logo1-eps-converted-to.pdf logo2.eps logo2.pdf logos.eps logos.pdf logos-eps-converted-to.pdf
cp main.pdf submission/main.pdf
# code release (no production data, no per-observation forecasts of the production groups)
cd ..
rm -rf code_release && mkdir -p code_release/results_real code_release/results_real_tempered code_release/results_sim code_release/results_sim_long code_release/results_sim_rev code_release/results_cv
cp bnnre_lib.py config.py prep_real.py cv_hyper.py run_real.py run_real_tempered.py run_seeds.sh run_temper_seeds.sh run_field_all.sh run_sim.py sim_batched.py run_sim_all.sh conformal.py make_results.py make_table1.py make_schematic.py build_all.sh code_release/
cp code_release_README.md code_release/README.md
cp out_real/summary_seed*.csv out_real/calib_seed*.csv code_release/results_real/
cp out_real_t/summary_seed*.csv code_release/results_real_tempered/
cp out_sim/sim_*_rep*.csv out_sim/calib_*.csv code_release/results_sim/
cp out_sim_long/sim_*_rep*.csv out_sim_long/calib_*.csv code_release/results_sim_long/
cp out_sim_rev/sim_*_rep*.csv out_sim_rev/calib_*.csv code_release/results_sim_rev/
cp out_cv/cv_results.csv code_release/results_cv/
cp frontiers/tables/key_numbers.json code_release/
rm -f frontiers/submission/code_release.zip; (cd code_release && zip -q -r ../frontiers/submission/code_release.zip .)
echo "built: $(date)"; pdfinfo frontiers/main.pdf | grep Pages; ls out_real/summary_seed*.csv | wc -l; ls out_real_t/summary_seed*.csv | wc -l; ls out_sim/sim_*_rep*.csv | wc -l; ls out_sim_long/sim_*_rep*.csv 2>/dev/null | wc -l; ls out_sim_rev/sim_*_rep*.csv 2>/dev/null | wc -l
