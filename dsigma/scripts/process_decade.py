import h5py
import numpy as np

from astropy.table import Table


def main():

    print("Reading in catalog data...")

    table_s = Table()

    with h5py.File('shear_catalog_sparse.hdf5') as fstream:
        for key in ['RA', 'DEC']:
            table_s[key] = fstream[key][:]
        for sheared in ['NOSHEAR', '1P', '1M', '2P', '2M']:
            for key in ['MCAL_G_1', 'MCAL_G_2', 'MCAL_W']:
                table_s[f'{key}_{sheared}'] = fstream[f'{key}_{sheared}'][:]
            table_s[f'MCAL_SEL_{sheared}'] = fstream[
                f'MCAL_SEL_{sheared}'][:].astype('int8')

    print("Calculating shear responses...")
    for i in [1, 2]:
        for j in [1, 2]:
            table_s[f'R_{i}{j}'] = ((table_s[f'MCAL_G_{i}_{j}P'] -
                                     table_s[f'MCAL_G_{i}_{j}M']) / (2 * 0.01))

    for sheared in ['1P', '1M', '2P', '2M']:
        table_s.remove_column(f'MCAL_G_1_{sheared}')
        table_s.remove_column(f'MCAL_G_2_{sheared}')

    # Determine whether galaxies are in the Northern Galactic Cap (NGC).
    ra_ngp, dec_ngp = 193, 27
    cos_alpha = (
        np.sin(np.deg2rad(table_s['DEC'])) * np.sin(np.deg2rad(dec_ngp)) +
        np.cos(np.deg2rad(table_s['DEC'])) * np.cos(np.deg2rad(dec_ngp)) *
        np.cos(np.deg2rad(table_s['RA'] - ra_ngp)))
    table_s['NGC'] = cos_alpha > 0

    print("Calculating selection responses...")
    for ngc in [True, False]:
        for z_bin in range(1, 5):
            use = (table_s['NGC'] == ngc) & (
                table_s['MCAL_SEL_NOSHEAR'] == z_bin)
            for i in range(1, 3):
                for j in range(1, 3):
                    e_p_ave = np.average(
                        table_s[f'MCAL_G_{i}_NOSHEAR'],
                        weights=table_s[f'MCAL_W_{j}P'] *
                        (table_s['NGC'] == ngc) *
                        (table_s[f'MCAL_SEL_{j}P'] == z_bin))
                    e_m_ave = np.average(
                        table_s[f'MCAL_G_{i}_NOSHEAR'],
                        weights=table_s[f'MCAL_W_{j}M'] *
                        (table_s['NGC'] == ngc) *
                        (table_s[f'MCAL_SEL_{j}M'] == z_bin))
                    table_s[f'R_{i}{j}'][use] += (e_p_ave - e_m_ave) / 0.02

    for key in ['MCAL_W', 'MCAL_SEL']:
        for sheared in ['1P', '1M', '2P', '2M']:
            table_s.remove_column(f'{key}_{sheared}')

    print("Mean responses...")
    for ngc in [True, False]:
        for z_bin in range(1, 5):
            use = (table_s['NGC'] == ngc) & (
                table_s['MCAL_SEL_NOSHEAR'] == z_bin)
            r = np.average((table_s['R_11'] + table_s['R_22']) * 0.5,
                           weights=table_s['MCAL_W_NOSHEAR'] * use)
            print(f"{'NGC' if ngc else 'SGC'} BIN {z_bin}: {r:.3f}")

    print("Adding multiplicative biases...")
    table_s['m'] = np.zeros(len(table_s))
    for ngc in [True, False]:
        use = table_s['NGC'] == ngc
        if ngc:
            m = np.array([-0.92, -1.90, -4.00, -3.73]) * 1e-2
        else:
            m = np.array([-1.33, -2.26, -3.67, -5.72]) * 1e-2
        table_s['m'][use] = np.where(
            table_s['MCAL_SEL_NOSHEAR'] > 0,
            m[table_s['MCAL_SEL_NOSHEAR'] - 1], np.nan)[use]

    # Only select galaxies suitable for cosmology.
    table_s = table_s[table_s['MCAL_SEL_NOSHEAR'] > 0]
    # Adjust redshift bin definition.
    table_s['MCAL_SEL_NOSHEAR'] -= 1

    keys = dict(ra='RA', dec='DEC', z_bin='MCAL_SEL_NOSHEAR',
                e_1='MCAL_G_1_NOSHEAR', e_2='MCAL_G_2_NOSHEAR',
                w='MCAL_W_NOSHEAR', m='m')
    for new_key, old_key in keys.items():
        table_s.rename_column(old_key, new_key)

    print("Writing data...")
    for ngc in [True, False]:

        fname = f"decade_{'ngc' if ngc else 'sgc'}.hdf5"

        use = table_s['NGC'] == ngc
        table_s[use].write(fname, path='catalog', exclude_names=['NGC'],
                           overwrite=True)

        table_n = Table()
        table_n['z'] = [z[1] for z in np.load('z_grid.npy')]
        table_n['n'] = np.column_stack(np.load(
            f"{'NGC' if ngc else 'SGC'}_n_of_z.npy"))
        table_n.write(fname, path='calibration', overwrite=True, append=True)

    print("Done!")


if __name__ == "__main__":
    main()
