"""Builds Table 1 (predictor fields) from the cleaned data.  Usage: python make_table1.py data.xlsx frontiers/tables/table_predictors_main.tex"""
import sys, numpy as np, pandas as pd
from prep_real import clean, load, RESPONSE

DESC = {  # field -> (description, unit)
 'week_num': ('Week of year of nursery close-out', 'weeks'),
 'Litters_female_year3': ('Litters per female per year (sow farm, rolling)', 'litters'),
 'mated_inventory_20wks3': ('Mated female inventory, 20-week average', 'head'),
 'PWMFyear3': ('Pigs weaned per mated female per year', 'pigs'),
 'nonproductive_days3': ('Non-productive sow days', 'days'),
 'number_services3': ('Number of services per week', 'services'),
 'wean_to_service3': ('Wean-to-first-service interval', 'days'),
 'abortions_rate3': ('Abortion rate', '\\%'),
 'farrow_sows3': ('Sows farrowed per week', 'sows'),
 'Total_born_avg3': ('Total born per litter', 'pigs'),
 'Stillborn_avg3': ('Stillborn rate', '\\%'),
 'Mummies_avg3': ('Mummified fetuses rate', '\\%'),
 'prenatal_losses_avg3': ('Prenatal losses per litter (total born minus born alive)', 'pigs'),
 'Born_alive_avg3': ('Born alive per litter', 'pigs'),
 'Gestation_days3': ('Gestation length', 'days'),
 'Interval_farrows3': ('Farrowing interval', 'days'),
 'Pre_weaning_mortality': ('Pre-weaning mortality', 'proportion'),
 'PWSow3': ('Pigs weaned per sow', 'pigs'),
 'productive_days_rate3': ('Productive days rate', '\\%'),
 'services_per_inventory_N_rate3': ('Services per inventory', '\\%'),
 'repeats__rate3': ('Repeat service rate', '\\%'),
 'gilts_bred_rate3': ('Gilts bred rate', '\\%'),
 'Last_week_wean_bred_rate3': ('Weaned sows bred within one week', '\\%'),
 'pregnant_105days_rate3': ('Pregnant at day 105 rate', '\\%'),
 'Cull_rate_annual3': ('Annualised sow cull rate', '\\%'),
 'Sow_Death_rate3': ('Annualised sow death rate', '\\%'),
 'avg_parity_at_farrow3': ('Average parity at farrowing', 'parity'),
 'Lactation_days3': ('Lactation length', 'days'),
 'final_inventory3': ('Final sow inventory', 'head'),
 'Farrowing__rate3': ('Farrowing rate', '\\%'),
 'NumberOfSources': ('Number of sow sources supplying the nursery group', 'count'),
 'TimeToFill': ('Days to fill the nursery site', 'days'),
 'StockingDensity1': ('Stocking density at placement', 'ft\\textsuperscript{2} per pig'),
 'MYCOatPlacement': ('\\textit{Mycoplasma hyopneumoniae} status of the sow source at placement', None),
 'PRRSatPlacement': ('PRRSV status of the sow source at placement', None),
 'PRRSVACCINE': ('PRRSV modified-live vaccine used at the sow farm', None),
 'WeanPigShot': ('Antimicrobial injection at weaning', None),
 'SowFarmMed': ('In-feed medication at the sow farm', None),
 'Ventilation1': ('Nursery ventilation type', None),
 'LateralCoronavirus': ('Lateral coronavirus introduction in the nursery', None),
 'LateralPRRS': ('Lateral PRRSV introduction in the nursery', None),
 'LateralEnteric': ('Lateral enteric pathogen introduction', None),
 'OtherHealthIssue': ('Other health issue recorded (APP)', None),
}

def tt(name):
    return '\\texttt{' + name.replace('_', '\\_\\allowbreak{}') + '}'

xlsx, dest = sys.argv[1], sys.argv[2]
dat, info = clean(xlsx)
train, blocks, linfo = load(xlsx)
caps = linfo['winsor_caps']
num_cols = [c for c in dat.select_dtypes(include=np.number).columns if c not in ('year_num', RESPONSE)]
cat_cols = [c for c in dat.select_dtypes(include=['object', 'string']).columns if c != 'SowSource']
n = dat.shape[0]
rows = []
for c in num_cols:
    s = dat[c].astype(float).clip(*caps[c])
    d, u = DESC[c]
    rows.append(f'{tt(c)} & {d} & {u}; mean {s.mean():.2f}, SD {s.std():.2f} & numerical \\\\')
for c in cat_cols:
    d, _ = DESC[c]
    vc = dat[c].value_counts()
    levels = '; '.join(f'{lv.replace("_", " ")} ({int(k)})' for lv, k in vc.items())
    rows.append(f'{tt(c)} & {d} & {levels} & categorical \\\\')
n_inputs = linfo['n_inputs']
cap = (f'Predictor fields recorded for each nursery group, as named in the production records, with descriptions, units or levels '
       f'(number of groups in parentheses) and summary statistics over the {n:,} groups after the unit correction and winsorisation described in the text. '
       f'Variables with the suffix 3 are rolling sow-farm indicators at the time of weaning; the four lateral-introduction and other-health-issue fields record events during the nursery phase. '
       f'One-hot encoding of the {len(cat_cols)} categorical fields gives {n_inputs} model inputs.')
out = ['\\begin{longtable}{p{4.3cm} p{5.5cm} p{4.8cm} p{1.5cm}}',
       '\\caption{' + cap + '}\\label{table:predictors}\\\\', '\\hline',
       'Variable (as recorded) & Description & Units or levels & Type \\\\', '\\hline', '\\endfirsthead',
       '\\multicolumn{4}{l}{\\small Table \\ref{table:predictors} (continued)}\\\\', '\\hline',
       'Variable (as recorded) & Description & Units or levels & Type \\\\', '\\hline', '\\endhead'] + rows + ['\\hline', '\\end{longtable}']
open(dest, 'w').write('\n'.join(out) + '\n')
print('wrote', dest, 'rows', len(rows), 'groups', n, 'inputs', n_inputs)
