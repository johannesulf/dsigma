import numpy as np
from astropy.table import Table, vstack


def main():

    print("Reading in catalog data...")

    fields = ['GAMA09H', 'GAMA15H', 'HECTOMAP', 'VVDS', 'WIDE12H', 'XMM']
    table_s = vstack([Table.read(f'{field}.fits.xz') for field in fields])
    # Remove regions with large B-modes.
    table_s = table_s[table_s['b_mode_mask'] == 1]

    keys = dict(
        ra='i_ra', dec='i_dec', z_bin='hsc_y3_zbin',
        e_1='i_hsmshaperegauss_e1', e_2='i_hsmshaperegauss_e2',
        w='i_hsmshaperegauss_derived_weight',
        m='i_hsmshaperegauss_derived_shear_bias_m',
        e_rms='i_hsmshaperegauss_derived_rms_e',
        R_2='i_hsmshaperegauss_resolution', mag_A='i_apertureflux_10_mag')

    for new_key, old_key in keys.items():
        table_s.rename_column(old_key, new_key)
    table_s.keep_columns(keys.keys())

    # HSC uses a different sign convention.
    table_s['e_2'] = -table_s['e_2']

    # Remove galaxies with bimodal P(z)'s.
    table_s = table_s[table_s['z_bin'] > 0]
    # Adjust redshift bin definition.
    table_s['z_bin'] = table_s['z_bin'].data.astype('int8') - 1

    print("Calculating selection bias...")
    d_R_2 = 0.01
    d_mag_A = 0.025
    # eq. (18) in 2304.00703
    table_s['m_sel'] = (
        0.01919 * (table_s['R_2'] < 0.3 + d_R_2) / d_R_2 -
        0.05854 * (table_s['mag_A'] > 25.5 - d_mag_A) / d_mag_A)
    table_s.remove_column('mag_A')

    print("Writing data...")
    table_s.write('hsc_y3.hdf5', path='catalog', overwrite=True)
    table_n = Table.read('nz.fits')
    # Create the columns expected by dsigma.
    table_n.rename_column('Z_MID', 'z')
    table_n['n'] = np.column_stack([table_n[f'BIN{i+1}'] for i in range(4)])
    table_n.keep_columns(['z', 'n'])
    table_n.write('hsc_y3.hdf5', path='calibration', overwrite=True,
                  append=True)

    print("Done!")


if __name__ == "__main__":
    main()
