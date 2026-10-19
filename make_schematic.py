"""Schematic of the BNN-RE ongoing forecasting procedure (Figure 1), drawn with matplotlib.
Usage: python make_schematic.py DEST_PNG DEST_TIF"""
import sys, textwrap, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

png, tif = sys.argv[1], sys.argv[2]
W, H = 180.0, 104.0                                   # mm
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
INK, GRID = '#0b0b0b', '#52514e'
C_DATA, C_FIT, C_FC, C_OBS = '#e9eef7', '#fde7dc', '#f1f0ed', '#f1f0ed'       # pale fills
C_RE, C_T, C_C, C_BNN = '#eb6834', '#1baf7a', '#eda100', '#2a78d6'

def box(x, y, w, h, text, fill, edge=GRID, fs=6.9, bold_first=False, lw=0.9, wrap=None, align='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6', fc=fill, ec=edge, lw=lw))
    lines = text.split('\n') if wrap is None else textwrap.wrap(text, wrap)
    if bold_first:
        ax.text(x + w / 2, y + h - 2.0, lines[0], ha='center', va='top', fontsize=fs + 0.3, fontweight='bold', color=INK)
        body = '\n'.join(lines[1:])
        ax.text(x + w / 2, y + h - 2.0 - 3.8, body, ha='center', va='top', fontsize=fs - 0.3, color=INK, linespacing=1.12)
    else:
        ax.text(x + w / 2, y + h / 2, '\n'.join(lines), ha='center', va='center', fontsize=fs, color=INK, linespacing=1.15)

def arrow(p, q, color=GRID, lw=1.1, style='-|>', rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0.5, shrinkB=0.5))

# ---------------------------------------------------------------- row 1: initial model and first forecast
y1, h1 = 80, 18
box(3, y1, 34, h1, 'Training records\n(groups nested in\nsow sources, 2019–2022)', C_DATA, bold_first=False)
box(44, y1, 44, h1, 'Initial BNN-RE fit\nmean-field variational inference gives\nthe approximate posterior\n$q_{\\phi}(\\theta,\\,u,\\,\\sigma_u,\\,\\sigma_e)$', C_FIT, bold_first=True)
box(95, y1, 38, h1, 'Forecast block 1\n5,000 posterior predictive draws\nper group; 95% interval\n[L, U] from their quantiles', C_FC, bold_first=True)
box(140, y1, 37, h1, 'Block 1 closes out\noutcomes $y$ observed;\ncoverage and width\nrecorded', C_OBS, bold_first=True)
arrow((37, y1 + h1 / 2), (44, y1 + h1 / 2)); arrow((88, y1 + h1 / 2), (95, y1 + h1 / 2)); arrow((133, y1 + h1 / 2), (140, y1 + h1 / 2))

# ---------------------------------------------------------------- row 2: the ongoing update cycle
y2, h2 = 46, 25
box(44, y2, 44, h2, 'Update #k  (k = 1, 2, …)\nrecycle the approximate posterior\nas the prior for $(\\theta, u, \\sigma_e)$;\nsource effects carried forward\nby identity, $\\sigma_u$ fixed;\nrefit by SVI on the new block(s)', C_FIT, edge=C_RE, lw=1.4, bold_first=True)
box(95, y2, 38, h2, 'Forecast block k+1\nintervals issued before\nits outcomes are known', C_FC, bold_first=True)
box(140, y2, 37, h2, 'Block k+1 closes out\noutcomes observed', C_OBS, bold_first=True)
arrow((88, y2 + h2 / 2), (95, y2 + h2 / 2)); arrow((133, y2 + h2 / 2), (140, y2 + h2 / 2))
# from block-1 outcomes down to Update #1, and the loop from block k+1 outcomes back to Update #k+1
arrow((158, y1), (158, y2 + h2 + 0.5), color=C_RE, lw=1.3);
ax.text(160, (y1 + y2 + h2) / 2 + 0.5, 'Update #1', fontsize=6.6, color=C_RE, va='center')
arrow((150, y2), (150, y2 - 5), color=C_RE, lw=1.3, style='-')
arrow((150, y2 - 5), (54, y2 - 5), color=C_RE, lw=1.3, style='-')
arrow((54, y2 - 5), (54, y2), color=C_RE, lw=1.3)
ax.text(90, y2 - 5.8, 'next update (k ← k + 1)', fontsize=6.6, color=C_RE, ha='center', va='top')

# schedules
box(3, y2 - 1, 34, h2 + 2, 'Two schedules\nConsecutive: prior = posterior\nof Update #k−1, data = block k\n\nCumulative: prior = initial\nposterior, data = blocks 1 … k', '#ffffff', edge=GRID, bold_first=True, fs=6.6)
arrow((37, y2 + h2 / 2), (44, y2 + h2 / 2), ls=(0, (3, 2)))

# ---------------------------------------------------------------- row 3: the two refinements
y3, h3 = 6, 25
box(44, y3, 44, h3, 'BNN-RE-T: tempered prior\nthe recycled prior of\n$\\eta_e=\\mathrm{logit}(\\sigma_e)$ keeps its\nlocation but counts as $n_0 = 60$\ngroups instead of $n_{prev}$, so the\nwidth follows the new period', '#e8f7f0', edge=C_T, lw=1.4, bold_first=True)
box(95, y3, 38, h3, 'BNN-RE-C: calibrated width\nscores $s_i=|y_i-c_i|/h_i$ from\nall blocks already forecast;\nnext intervals $c \\pm \\hat q\\,h$ with\n$\\hat q$ their 96% conformal\nquantile', '#fdf4df', edge=C_C, lw=1.4, bold_first=True)
arrow((74, y3 + h3), (74, y2), color=C_T, lw=1.3, ls=(0, (3, 2)))
arrow((114, y3 + h3), (114, y2), color=C_C, lw=1.3, ls=(0, (3, 2)))
box(3, y3, 34, h3, 'Refinements\n(used one at a time)\nT acts inside the update\nand re-learns the residual\nvariance; C acts on the\nissued intervals and\nneeds no refitting', '#ffffff', edge=GRID, bold_first=True, fs=6.9)
# scores feed back from observed outcomes to the calibration
arrow((170, y2), (170, y3 + h3 / 2), color=C_C, lw=1.0, ls=(0, (3, 2)), style='-')
arrow((170, y3 + h3 / 2), (133, y3 + h3 / 2), color=C_C, lw=1.0, ls=(0, (3, 2)))
ax.text(171, (y2 + y3 + h3 / 2) / 2, 'scores', fontsize=6.4, color=C_C, va='center')

fig.savefig(png, dpi=300)
Image.open(png).convert('RGB').save(tif, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
print('wrote', png, tif)
