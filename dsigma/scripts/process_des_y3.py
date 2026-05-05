import h5py
import numpy as np
from astropy.table import Table


def main():

    print("Reading in catalog data...")

    table_s = Table()

    with h5py.File('DESY3_sompz_v0.50.h5') as fstream:
        table_s['z_bin'] = fstream[
            'catalog/sompz/unsheared/bhat'][:].astype('int8')
        for sheared in ['1m', '1p', '2m', '2p']:
            table_s[f'z_bin_{sheared}'] = fstream[
                f'catalog/sompz/sheared_{sheared}/bhat'][:].astype('int8')

    with h5py.File('DESY3_metacal_v03-004.h5') as fstream:
        for key in ['ra', 'dec', 'e_1', 'e_2', 'R11', 'R12', 'R21', 'R22',
                    'weight']:
            table_s[key] = fstream['catalog/unsheared/' + key][:]
        for sheared in ['1m', '1p', '2m', '2p']:
            table_s[f'weight_{sheared}'] = fstream[
                f'catalog/sheared_{sheared}/weight'][:]

    with h5py.File('DESY3_indexcat.h5') as fstream:
        for flag in ['select', 'select_1p', 'select_1m', 'select_2p',
                     'select_2m']:
            table_s[flag] = np.zeros(len(table_s), dtype=bool)
            table_s[flag][fstream['index/' + flag][()]] = True

    print("Calculating selection responses...")
    for z_bin in range(4):
        use = table_s['z_bin'] == z_bin
        for i in range(1, 3):
            for j in range(1, 3):
                use_p = (table_s[f'select_{j}p'] &
                         (table_s[f'z_bin_{j}p'] == z_bin))
                e_p_ave = np.average(
                    table_s[f'e_{i}'][use_p],
                    weights=table_s[f'weight_{j}p'][use_p])
                use_m = (table_s[f'select_{j}m'] &
                         (table_s[f'z_bin_{j}m'] == z_bin))
                e_m_ave = np.average(
                    table_s[f'e_{i}'][use_m],
                    weights=table_s[f'weight_{j}m'][use_m])
                table_s[f'R{i}{j}'][use] += (e_p_ave - e_m_ave) / 0.02

    for key in ['select', 'weight', 'z_bin']:
        for sheared in ['1m', '1p', '2m', '2p']:
            table_s.remove_column(f'{key}_{sheared}')

    # Only select galaxies suitable for cosmology.
    table_s = table_s[table_s['weight'] > 0]
    table_s = table_s[table_s['z_bin'] >= 0]
    table_s = table_s[table_s['select']]
    table_s.remove_column('select')

    print("Adding multiplicative biases...")
    m = np.array([-0.63, -1.98, -2.41, -3.69]) * 1e-2
    table_s['m'] = m[table_s['z_bin']]

    keys = dict(w='weight', R_11='R11', R_22='R22', R_12='R12', R_21='R21')
    for new_key, old_key in keys.items():
        table_s.rename_column(old_key, new_key)

    print("Writing data...")
    table_s.write('des_y3.hdf5', path='catalog', overwrite=True)

    table_n = Table.read(
        '2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits', hdu=6)
    table_n.rename_column('Z_MID', 'z')
    table_n['n'] = np.column_stack([table_n[f'BIN{i+1}'] for i in range(4)])
    table_n.keep_columns(['z', 'n'])
    table_n.write('des_y3.hdf5', path='calibration', overwrite=True,
                  append=True)

    print("Done!")


if __name__ == "__main__":
    main()
