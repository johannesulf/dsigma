import argparse
import os

import numpy as np
from astropy.cosmology import Planck15
from astropy.table import Table, vstack

from dsigma.precompute import precompute
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal around BOSS galaxies.')

parser.add_argument('survey', help='the lensing survey')
args = parser.parse_args()

cosmology = Planck15
rp_bins = np.logspace(-1, 1.6, 14)
z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

table_l = vstack([Table.read('galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                  Table.read('galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')])
table_r = vstack([Table.read('random0_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                  Table.read('random0_DR12v5_CMASSLOWZTOT_North.fits.gz')])
keys = dict(z='Z', ra='RA', dec='DEC')
for table in [table_l, table_r]:
    for new_key, old_key in keys.values():
        table.rename_column(old_key, new_key)
    table.keep_columns(keys.keys())
    table['w_sys'] = 1

table_l = table_l[table_l['z'] >= np.amin(z_bins)]
table_r = table_r[table_r['z'] >= np.amin(z_bins)][::5]

if args.survey.lower() == 'decade':

    table_s = Table.read('decade_ngc.hdf5', path='catalog')
    table_n = Table.read('decade_ngc.hdf5', path='calibration')

    table_s['z'] = np.array([0.0, 0.381, 0.619, 0.803])[table_s['z_bin']]

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.1)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           matrix_shear_response_correction=True)

elif args.survey.lower() == 'des':

    table_s = Table.read('des_y3.hdf5', path='catalog')
    table_n = Table.read('des_y3.hdf5', path='calibration')

    table_s['z'] = np.array([0.0, 0.358, 0.631, 0.872])[table_s['z_bin']]

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.1)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           matrix_shear_response_correction=True)

elif args.survey.lower() == 'hsc':

    table_s = Table.read('hsc_y3.fits')
    # Remove regions with large B-modes.
    table_s = table_s[table_s['b_mode_mask'] == 1]
    table_s = dsigma_table(table_s, 'source', survey='HSC')
    table_s['m_sel'] = hsc.multiplicative_selection_bias(table_s)
    # Remove galaxies with bimodal P(z)'s.
    table_s = table_s[table_s['z_bin'] > 0]
    # dsigma expects the first redshift bin to be 0, not 1.
    table_s['z_bin'] = table_s['z_bin'] - 1

    table_n = Table.read('nz.fits')
    # Create the columns expected by dsigma.
    table_n.rename_column('Z_MID', 'z')
    table_n['n'] = np.column_stack([table_n[f'BIN{i+1}'] for i in range(4)])
    table_n.keep_columns(['z', 'n'])

    table_s['z'] = np.sum(table_n['z'][:, np.newaxis] *
                          table_n['n'], axis=0)[table_s['z_bin']]

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.3)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           shear_responsivity_correction=True,
                           selection_bias_correction=True)

elif args.survey.lower() == 'kids':

    table_s = Table.read('kids_legacy.hdf5', path='catalog')
    table_n = Table.read('kids_legacy.hdf5', path='calibration')

    table_s['z'] = np.array([0.1, 0.42, 0.58, 0.71, 0.90, 1.14])[
        table_s['z_bin']]

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.1)
    stacking_kwargs = dict(scalar_shear_response_correction=True)

else:
    raise ValueError("Survey must be 'des', 'hsc' or 'kids'.")

precompute_kwargs.update(dict(
    n_jobs=os.cpu_count(), comoving=True, cosmology=cosmology,
    progress_bar=True))

# Pre-compute the signal.
precompute(table_l, table_s, rp_bins, **precompute_kwargs)
precompute(table_r, table_s, rp_bins, **precompute_kwargs)

# Drop all lenses and randoms that did not have any nearby source.
table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
table_r = table_r[np.sum(table_r['sum 1'], axis=1) > 0]

centers = compute_jackknife_fields(
    table_l, 100, weights=np.sum(table_l['sum 1'], axis=1))
compute_jackknife_fields(table_r, centers)

# Stack the signal.
stacking_kwargs['random_subtraction'] = True

for lens_bin in range(len(z_bins) - 1):
    use_l = ((z_bins[lens_bin] <= table_l['z']) &
             (table_l['z'] < z_bins[lens_bin + 1]))
    use_r = ((z_bins[lens_bin] <= table_r['z']) &
             (table_r['z'] < z_bins[lens_bin + 1]))

    stacking_kwargs['table_r'] = table_r[use_r]
    stacking_kwargs['return_table'] = True
    result = excess_surface_density(table_l[use_l], **stacking_kwargs)
    stacking_kwargs['return_table'] = False
    result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
        excess_surface_density, table_l[use_l], **stacking_kwargs)))

    result.write(f'{args.survey.lower()}_{lens_bin}.csv', overwrite=True)
