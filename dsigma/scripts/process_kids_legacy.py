import numpy as np
from astropy.table import Table


def main():

    print("Reading in catalog data...")

    table_s = Table.read('KiDS_Legacy_NS_unblind_final.fits.gz')

    keys = dict(ra='RAJ2000', dec='DECJ2000', z_bin='TOMOBIN', e_1='e1',
                e_2='e2', w='weight')
    for new_key, old_key in keys.items():
        table_s.rename_column(old_key, new_key)
    table_s.keep_columns(keys.keys())

    # Adjust redshift bin definition.
    table_s['z_bin'] = table_s['z_bin'].data.astype('int8') - 1

    print("Adding multiplicative biases...")
    m = np.array([-2.3, -1.6, -1.1, 2.0, 3.0, 4.5]) * 1e-2
    table_s['m'] = m[table_s['z_bin']]

    print("Computing redshift distributions...")
    z_bins = np.linspace(0, 10, 1001)
    table_n = Table()
    table_n['z'] = 0.5 * (z_bins[1:] + z_bins[:-1])
    table_n['n'] = np.zeros((len(table_n), 6))

    table_kidz = Table.read('KiDZ_Legacy_unblind_final.fits')
    table_kidz['z_bin'] = table_kidz['TOMOBIN'] - 1
    for z_bin in range(6):
        offset = [-0.026, 0.014, -0.002, 0.008, -0.011, -0.054][z_bin]
        use = (table_kidz['TOMOBIN'] - 1) == z_bin
        z = table_kidz['z_spec'][use] - offset
        # Ensure all values fall within the bins.
        z = np.maximum(z, np.nextafter(np.amin(z_bins), np.inf))
        z = np.minimum(z, np.nextafter(np.amax(z_bins), -np.inf))
        w = table_kidz['final_weight'][use]
        table_n['n'][:, z_bin] = np.histogram(z, z_bins, weights=w)[0]
        table_n['n'][:, z_bin] /= np.sum(table_n['n'][:, z_bin])
        z_ave = np.average(table_n['z'], weights=table_n['n'][:, z_bin])
        print(f"KiDS-{z_bin}: <z> = {z_ave:.3f}")

    print("Writing data...")
    table_s.write('kids_legacy.hdf5', path='catalog', overwrite=True)
    table_n.write('kids_legacy.hdf5', path='calibration', overwrite=True,
                  append=True)

    print("Done!")


if __name__ == "__main__":
    main()
