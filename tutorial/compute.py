import argparse
import numpy as np
import multiprocessing
from astropy.table import Table, vstack, join
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from dsigma.surveys import des, hsc, kids
from astropy.cosmology import Planck15

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal around BOSS galaxies.')

parser.add_argument('survey', help='the lensing survey')
args = parser.parse_args()

cosmology = Planck15
rp_bins = np.logspace(-1, 1.6, 14)
z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

table_l = vstack([Table.read('galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                  Table.read('galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')])
table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC',
                       w_sys=1)
table_l = table_l[table_l['z'] >= np.amin(z_bins)]

table_r = vstack([Table.read('random0_DR12v5_CMASSLOWZTOT_South.fits.gz'),
                  Table.read('random0_DR12v5_CMASSLOWZTOT_North.fits.gz')])
table_r = dsigma_table(table_r, 'lens', z='Z', ra='RA', dec='DEC',
                       w_sys=1)[::5]
table_r = table_r[table_r['z'] >= np.amin(z_bins)]

if args.survey.lower() == 'des':

    table_s = Table.read('des_y3.hdf5', path='catalog')
    table_s = dsigma_table(table_s, 'source', survey='DES')

    table_s['m_sel'] = np.zeros(len(table_s))
    for z_bin in range(4):
        select = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[select])
        print(f"Bin {z_bin + 1}: m_sel = "
              f"{100 * 0.5 * np.sum(np.diag(R_sel)):.1f}%")
        table_s['m_sel'][select] = 0.5 * np.sum(np.diag(R_sel))

    table_s = table_s[table_s['z_bin'] >= 0]
    table_s = table_s[table_s['flags_select']]
    table_s['m'] = des.multiplicative_shear_bias(
        table_s['z_bin'], version='Y3')

    table_n = Table.read('des_y3.hdf5', path='redshift')
    table_s['z'] = np.array([0.0, 0.358, 0.631, 0.872])[table_s['z_bin']]

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.1)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           matrix_shear_response_correction=True,
                           selection_bias_correction=True)

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

    table_s = Table.read('KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits')
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(
        table_s['z'], version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(
        table_s['z_bin'], version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]
    table_s['z'] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[table_s['z_bin']]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_' +
             'v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    table_n = Table()
    table_n['z'] = np.genfromtxt(fname.format(1))[:, 0] + 0.025
    table_n['n'] = np.column_stack(
        [np.genfromtxt(fname.format(i + 1))[:, 1] for i in range(5)])

    precompute_kwargs = dict(table_n=table_n, lens_source_cut=0.1)
    stacking_kwargs = dict(scalar_shear_response_correction=True)

else:
    raise ValueError("Survey must be 'des', 'hsc' or 'kids'.")

precompute_kwargs.update(dict(
    n_jobs=4, comoving=True, cosmology=cosmology, progress_bar=True))

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
    mask_l = ((z_bins[lens_bin] <= table_l['z']) &
              (table_l['z'] < z_bins[lens_bin + 1]))
    mask_r = ((z_bins[lens_bin] <= table_r['z']) &
              (table_r['z'] < z_bins[lens_bin + 1]))

    stacking_kwargs['table_r'] = table_r[mask_r]
    stacking_kwargs['return_table'] = True
    result = excess_surface_density(table_l[mask_l], **stacking_kwargs)
    stacking_kwargs['return_table'] = False
    result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
        excess_surface_density, table_l[mask_l], **stacking_kwargs)))

    result.write(f'{args.survey.lower()}_{lens_bin}.csv', overwrite=True)
