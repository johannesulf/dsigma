import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

survey_list = ['DECADE', 'DES', 'HSC', 'KiDS']
color_list = ['orange', 'red', 'royalblue', 'purple']
offset_list = [0, 0.05, 0.1, 0.15]

for survey, color, offset in zip(survey_list, color_list, offset_list):
    table = Table.read('{}_0.csv'.format(survey.lower()))

    rp = table['rp']

    plt.errorbar(
        rp * np.exp(offset), rp * table['ds'], yerr=table['ds_err'] * rp,
        fmt='.', color=color, label=survey, zorder=offset*100
    )[2][0].set_capstyle('round')

plt.axhline(0, ls='--', color='black')
plt.legend(loc='upper center', fancybox=True, shadow=True, ncols=2,
           handletextpad=0,
           columnspacing=0)
plt.xscale('log')

plt.xlabel(r'Projected Radius $r_p \, [\mathrm{Mpc}]$')
plt.ylabel(r'ESD $r_p \times \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.7)
plt.savefig('plot.png', dpi=600)
