import fitsio
import argparse
import numpy as np
import multiprocessing
from astropy.table import Table, vstack, hstack, join
from dsigma.helpers import dsigma_table
from dsigma.precompute import add_maximum_lens_redshift, add_precompute_results
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from dsigma.surveys import des, kids
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

    table_s = []

    fname_list = ['mcal-y1a1-combined-riz-unblind-v4-matched.fits',
                  'y1a1-gold-mof-badregion_BPZ.fits',
                  'mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits']
    columns_list = [['e1', 'e2', 'R11', 'R12', 'R21', 'R22', 'ra', 'dec',
                     'flags_select', 'flags_select_1p', 'flags_select_1m',
                     'flags_select_2p', 'flags_select_2m'], ['Z_MC'],
                    ['MEAN_Z']]

    for fname, columns in zip(fname_list, columns_list):
        table_s.append(Table(fitsio.read(fname, columns=columns),
                             names=columns))

    table_s = hstack(table_s)
    table_s = dsigma_table(table_s, 'source', survey='DES')
    table_s['z_bin'] = des.tomographic_redshift_bin(table_s['z'])

    for z_bin in range(4):
        use = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[use])
        table_s['R_11'][use] += 0.5 * np.sum(np.diag(R_sel))
        table_s['R_22'][use] += 0.5 * np.sum(np.diag(R_sel))

    table_s = table_s[(table_s['flags_select'] == 0) &
                      (table_s['z_bin'] != -1)]

    table_c = table_s['z', 'z_true', 'w']
    table_c['w_sys'] = 0.5 * (table_s['R_11'] + table_s['R_22'])

    precompute_kwargs = {'table_c': table_c}
    stacking_kwargs = {'tensor_shear_response_correction': True,
                       'photo_z_dilution_correction': True}

elif args.survey.lower() == 'hsc':

    table_s = Table.read('hsc_s16a_lensing.fits')
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([
        Table.read('pdf-s17a_wide-9812.cat.fits'),
        Table.read('pdf-s17a_wide-9813.cat.fits')])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read('Afterburner_reweighted_COSMOS_photoz_FDFC.fits')
    table_c_2.rename_column('S17a_objid', 'id')
    table_c = join(table_c_1, table_c_2, keys='id')
    table_c = dsigma_table(table_c, 'calibration', w_sys='SOM_weight',
                           w='weight_source', z_true='COSMOS_photoz',
                           survey='HSC')

    precompute_kwargs = {'table_c': table_c}
    stacking_kwargs = {'scalar_shear_response_correction': True,
                       'shear_responsivity_correction': True,
                       'photo_z_dilution_correction': True,
                       'hsc_selection_bias_correction': True}

elif args.survey.lower() == 'kids':

    table_s = Table.read('KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits')
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(
        table_s['z'], version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'], version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]
    table_s['z'] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[table_s['z_bin']]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_' +
             'v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    table_n = Table()
    table_n['z'] = np.genfromtxt(fname.format(1))[:, 0] + 0.025
    table_n['n'] = np.vstack(
        [np.genfromtxt(fname.format(i + 1))[:, 1] for i in range(5)]).T

    precompute_kwargs = {'table_n': table_n}
    stacking_kwargs = {'scalar_shear_response_correction': True}

else:
    raise ValueError("Survey must be 'des', 'hsc' or 'kids'.")

add_maximum_lens_redshift(table_s, dz_min=0.1)
if 'table_c' in precompute_kwargs.keys():
    add_maximum_lens_redshift(table_c, dz_min=0.1)

precompute_kwargs.update({
    'n_jobs': multiprocessing.cpu_count(), 'comoving': True,
    'cosmology': cosmology})

# Pre-compute the signal.
add_precompute_results(table_l, table_s, rp_bins, **precompute_kwargs)
add_precompute_results(table_r, table_s, rp_bins, **precompute_kwargs)

# Add jackknife fields.
table_l['n_s_tot'] = np.sum(table_l['sum 1'], axis=1)
table_l = table_l[table_l['n_s_tot'] > 0]

table_r['n_s_tot'] = np.sum(table_r['sum 1'], axis=1)
table_r = table_r[table_r['n_s_tot'] > 0]

add_continous_fields(table_l, distance_threshold=2)
centers = jackknife_field_centers(table_l, 100, weight='n_s_tot')
add_jackknife_fields(table_l, centers)
add_jackknife_fields(table_r, centers)

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

    result.write('{}_{}.csv'.format(args.survey.lower(), lens_bin))
