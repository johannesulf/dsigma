import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

survey_list = ['DES', 'HSC', 'KiDS']
color_list = ['red', 'royalblue', 'purple']
offset_list = [0, 0.15, 0.30]

for survey, color, offset in zip(survey_list, color_list, offset_list):
    table = Table.read('{}_0.csv'.format(survey.lower()))

    rp = table['rp']

    plotline, caps, barlinecols = plt.errorbar(
        rp * np.exp(offset), rp * table['ds'], yerr=table['ds_err'] * rp,
        fmt='.', color=color, label=survey)
    plt.setp(barlinecols[0], capstyle='round')

plt.axhline(0, ls='--', color='black')
plt.legend(loc='best', frameon=False)
plt.xscale('log')

plt.xlabel(r'Projected Radius $r_p \, [\mathrm{Mpc}]$')
plt.ylabel(r'ESD $r_p \times \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.8)
plt.savefig('plot.png', dpi=300)
