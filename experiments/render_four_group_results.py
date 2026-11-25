"""Render a saved four-group summary without making model requests."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
groups = json.loads((args.run_dir / 'summary.json').read_text(encoding='utf-8'))['groups']
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
x = np.arange(len(groups))
for index, (metric, label) in enumerate([
    ('valid_rate', 'Validity'), ('novelty_rate', 'Uniqueness'), ('recovery_rate', 'Recovery')
]):
    axes[0].bar(x + (index - 1) * .24,
                [g[metric]['mean'] * 100 for g in groups.values()], .22,
                yerr=[g[metric]['std'] * 100 for g in groups.values()], capsize=3, label=label)
axes[0].set_xticks(x, list(groups))
axes[0].set_ylim(0, 125)
axes[0].set_ylabel('Percent (SD is not a confidence interval)')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].set_title('One instance, three seeds (mean +/- SD)')
bars = axes[1].bar(list(groups), [g['seconds'] for g in groups.values()],
                   color=['#777777', '#377eb8', '#4daf4a', '#984ea3'])
axes[1].bar_label(bars, fmt='%.1f s', padding=3)
axes[1].set_ylim(0, 680)
axes[1].set_ylabel('Total seconds for 24 requests')
axes[1].set_title('Measured time, including loading')
fig.savefig(args.run_dir / 'comparison.png', dpi=170)
plt.close(fig)
