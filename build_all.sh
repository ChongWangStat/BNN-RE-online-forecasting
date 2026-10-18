#!/bin/bash
# Rebuild tables, figures, the Frontiers manuscript, the LaTeX source zip and the code release from the saved run outputs.
# Inputs: out_real_jit (BNN, BNN-RE field runs), out_real_t60 (BNN-RE-T field runs), out_sim2 (5-block simulations),
#         out_sim_long (21-block simulations).  BNN-RE-C is computed post hoc from the saved per-observation predictions.
set -e
cd /home/claude/bnnre
python3 -W ignore make_results.py out_real_jit out_real_t60 out_sim2 out_sim_long frontiers/tables frontiers/submission/figures > frontiers/tables/key_numbers.log
python3 make_schematic.py frontiers/figures/Figure_1.png frontiers/submission/figures/Figure_1.tif > /dev/null
cd frontiers
[ -f tables/sim_long_table.tex ] || printf '\\begin{tabular}{l}\\hline long-horizon simulation results pending \\\\ \\hline\\end{tabular}\n' > tables/sim_long_table.tex
pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
bibtex main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main > build_main.log 2>&1 || true
grep -E "^!" build_main.log || echo "LaTeX: no errors"
grep -i "undefined\|multiply" build_main.log || echo "LaTeX: no undefined references"
rm -f submission/latex_source.zip
zip -q -r submission/latex_source.zip main.tex refs.bib FrontiersinVancouver.cls Frontiers-Vancouver.bst tables figures logo1.eps logo1.pdf logo1-eps-converted-to.pdf logo2.eps logo2.pdf logos.eps logos.pdf logos-eps-converted-to.pdf
cp main.pdf submission/main.pdf
# code release (no production data)
cd /home/claude/bnnre
rm -rf code_release && mkdir -p code_release/results_real code_release/results_real_tempered code_release/results_sim code_release/results_sim_long
cp bnnre_lib.py prep_real.py run_real.py run_real_tempered.py run_seeds.sh run_temper_seeds.sh run_sim.py sim_batched.py run_sim_rest.sh conformal.py analyze_improvements.py make_results.py make_schematic.py build_all.sh code_release/
cp code_release_README.md code_release/README.md
cp out_real_jit/summary_seed*.csv code_release/results_real/ 2>/dev/null || true
cp out_real_t60/summary_seed*.csv code_release/results_real_tempered/ 2>/dev/null || true
cp out_sim2/sim_*_rep*.csv code_release/results_sim/ 2>/dev/null || true
cp out_sim_long/sim_*_rep*.csv code_release/results_sim_long/ 2>/dev/null || true
cp frontiers/tables/key_numbers.json code_release/
rm -f frontiers/submission/code_release.zip; (cd code_release && zip -q -r ../frontiers/submission/code_release.zip .)
echo "built: $(date)"; pdfinfo frontiers/main.pdf | grep Pages; ls out_real_jit/summary_seed*.csv | wc -l; ls out_real_t60/summary_seed*.csv | wc -l; ls out_sim2/*.csv | wc -l; ls out_sim_long/*.csv 2>/dev/null | wc -l
