"""Useful functions for dsigma pipeline."""

import copy

import numpy as np
from numpy.lib.recfunctions import append_fields

from scipy import ndimage
from scipy.optimize import curve_fit

from astropy import units, constants

import smatch

__all__ = ["CSQUARE_OVER_4PIG", "mpc_per_degree", "get_matches",
           "projection_angle", "get_radial_bins", "sigma_crit",
           "get_radial_bin_centers", "get_cosmology_arry",
           "random_downsampling", "random_assign_redshift",
           "bootstrap_fields", 'corr2cov', 'smooth_cov', 'fit_avg_ratio']

CSQUARE_OVER_4PIG = (
    constants.c ** 2 /
    (4 * np.pi * constants.G)).to(units.Msun / units.pc).value

#CEE = 3.0E5
#GEE = 4.2994E-9
#CSQUARE_OVER_4PIG = (CEE ** 2.0) / (np.pi * 4.0 * GEE * 1.0E6)


def bootstrap_fields(fields):
    """Estimate the average SMF using bootstrap fields.

    Parameter
    ---------
    fields : list or array
        List or array of indices to bootstrap from.

    Return
    ------
        A bootstrap resampled array of indices.
    """
    return np.random.choice(fields, len(fields), replace=True)


def mpc_per_degree(cosmo, redshift, comoving=False):
    """Estimate the angular scale in Mpc/degree at certain redshift.

    Parameters
    ----------
    cosmo : cosmology.Cosmology object
        Cosmology object from the `cosmology` package by Erin Sheldon.
    redshift : float
        Redshift of the object.
    comoving : boolen
        Use comoving distance instead of physical distance when True.
        Default: False

    Return
    ------
        Physical scale in unit of Mpc/degree.
    """
    # cosmo.Da returns Mpc/rad and deg2rad is effectively (rad/deg)*arg
    if comoving:
        return cosmo.Dc(0.0, redshift) * np.deg2rad(1)

    return cosmo.Da(0.0, redshift) * np.deg2rad(1)


def get_matches(ra1, dec1, ra2, dec2, mpc_deg=None, scale='physical',
                rmin=0.1, rmax=20.0, nside=64):
    """Cross-match coordinates.

    Parameters
    ----------
    ra1, dec1 : float or numpy array
        Coordinates of the first group of objects.
    ra2, dec2 : float or numpy array
        Coordinates of the second group of objects.
    rmin : float, optional
        Minimum radius limit. Default: 0.1 Mpc.
    rmax : float, optional
        Maximum radius limit. Default: 20.0 Mpc.
    mpc_deg : float
        Phyiscal size in unit of Mpc per degree. Default: None.
    scale : string
        Coordinate type, `physical` or `angular`.  Default: `physical`.
    nside : int
        Default: 64

    Return
    ------
        Boolen array for matched objects.

    """
    if scale not in ["physical", "angular"]:
        raise Exception("Scale should be either physical or angular")

    max_radius, min_radius = rmax, rmin
    if scale == 'physical':
        if mpc_deg is None:
            raise Exception("Please provide correct redshift information")
        # Physical match: max_radius is in Mpc.
        max_degrees = (max_radius / mpc_deg)
    else:
        # Angular match: max_radius is already in unit of degree
        max_degrees = max_radius

    # TODO: larger nside makes the matching process slower,
    matches = smatch.match(ra1, dec1, max_degrees, ra2, dec2,
                           nside=nside, maxmatch=0)

    # this is a bit dirty... But I don't think it is terrible.
    matches.dtype.names = ('i1', 'i2', 'dist')
    matches['dist'] = np.rad2deg(np.arccos(matches['dist']))

    # Convert the distances from angular unit into physical unit if necessary.
    if scale == 'physical':
        if isinstance(mpc_deg, np.ndarray):
            matches['dist'] *= mpc_deg[matches['i1']]
        else:
            matches['dist'] *= mpc_deg

    # Find the matched ones between min_radius and max_radius
    mask = (matches['dist'] > min_radius) & (matches['dist'] <= max_radius)

    return matches[mask]


def projection_angle(lensra, lensdec, sourcera, sourcedec):
    """Calculate projection angle between lens and sources.

    Parameters
    ----------
    lensra, lensdec : float or numpy array
        Coordinates of the lenses.
    sourcera, sourcedec : float or numpy array
        Coordinates of the background sources.

    Return
    ------
    cos_2phi, sin_2phi : float or numpy array
        The cos() and sin() of r'2 \times \phi' angle.
    """
    # Convert everything into radian.
    ra_l, dec_l = np.deg2rad(lensra), np.deg2rad(lensdec)
    ra_s, dec_s = np.deg2rad(sourcera), np.deg2rad(sourcedec)

    # Calculating the tan(phi)
    tan_phi = ((np.cos(dec_l) * np.sin(dec_s) -
                np.sin(dec_l) * np.cos(dec_s) * np.cos(ra_s - ra_l)) /
               (np.cos(dec_s) * np.sin(ra_s - ra_l)))

    cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
    sin_2phi = (2.0 * tan_phi / (1.0 + tan_phi * tan_phi))

    # This throws warnings where cosbd*sinfbra==0.
    # We overwrite it later so it's not a problem;
    # I just haven't figured out how to turn the warnings off.
    inf_tan_phi = (np.cos(dec_s) * np.sin(ra_s - ra_l) == 0)
    if isinstance(inf_tan_phi, np.ndarray):
        cos_2phi[inf_tan_phi] = -1.
        sin_2phi[inf_tan_phi] = 0.
    elif inf_tan_phi:
        cos_2phi = -1.
        sin_2phi = 0.

    return cos_2phi, sin_2phi


def get_radial_bins(dist, rmin=0.1, rmax=20.0, nbins=11):
    """Create radial bins.

    Parameters
    ----------
    dist : numpy array
        Angular distances
    rmin : float, optional
        Mininmum radius.  Default: 0.1 Mpc
    rmax : float, optional
        Maximum radius. Default: 20.0 Mpc
    nbins : int, optional
        Number of radial bins. Default: 11

    Return
    ------
        Numpy array for indexes of radial bins.
    """

    return np.floor(
        (np.log(dist) - np.log(rmin)) * nbins / np.log(rmax / rmin)
        ).astype(int)


def sigma_crit(zl, zs, cosmo, comoving=False):
    """Calculate the distance term in Sigma_crit.

    Convert it into pc^1

    Parameter
    ---------
    zl : float or numpy array
        Redshift of lens.
    zs : float or numpy array
        Redshift of source.
    cosmos : cosmology.Cosmology object
        Cosmology object from `cosmology` package by Erin Sheldon.
    comoving : boolen, optional
        Flag for using comoving instead of physical unit. Default: False

    Return
    ------
        Critical surface density measurements

    """
    dist_term = ((1e-6 * cosmo.Da(0, zs) /
                  (cosmo.Da(zl, zs) * cosmo.Da(0, zl))))

    if comoving:
        return CSQUARE_OVER_4PIG * dist_term * (1.0 / (1. + zl)) ** 2

    return CSQUARE_OVER_4PIG * dist_term


def get_radial_bin_centers(cfg_binning):
    """Return radial bin centers.

    Helpers that return values we want to store with the data

    Parameters
    ----------
    cfg_binning : dict
        Dictionary for radial binning parameters.

    Return
    ------
        1-D array for the center of each bins.
    """
    # TODO: Figure out why we use log and exp instead of log10 and 10^.
    rmin, rmax = np.log(cfg_binning["rmin"]), np.log(cfg_binning["rmax"])
    nbins = cfg_binning["nbins"]
    step = (rmax - rmin) / nbins

    return np.array([np.exp(rmin + step / 2.0 + n * step) for n in range(nbins)])


def get_cosmology_arry(cosmo):
    """Return an array that describes the cosmology parameters.

    Parameters
    ----------
    cosmo : cosmology.cosmology.Cosmo
        Cosmology models

    Return
    ------
    cosmo_arr : numpy 1-D array
        Array that describes key cosmology parameters.
    """
    keys = ["omega_m", "omega_l", "omega_k", "flat", "H0"]

    return np.array([getattr(cosmo, i)() for i in keys])


def random_downsampling(random_cat, n_random):
    """Downsample the random catalog to certain number.

    Parameters
    ----------
    random_cat : numpy array
        Random catalog.
    n_random : int
        Number of required random objects.

    Return
    ------
        Downsampled random catalog.
    """
    if n_random < len(random_cat):
        random_downsample = np.random.choice(random_cat, n_random)
    else:
        print("# N_random number is too large!")
        random_downsample = random_cat

    return random_downsample


def random_assign_redshift(random_cat, lens_cat=None, z_col='z',
                           z_min=0.01, z_max=1.0):
    """Assign redshifts to the random catalog throught random draw from
    a lens catalog, or use flat distribution between a lower and higher limits.

    Parameters
    ----------
    random_cat : numpy array
        Random catalog.
    lens_cat : numpy array, optional
        Lenses catalog. Default: None
    z_col : str, optional
        Name of the redshift column in the lens catalog. Default: 'z'
    z_min: float, optional
        Lower redshift limit. Default: 0.01
    z_max: float, optional
        Higher redshift limit.  Default: 1.0

    Return
    ------
        Random catalog with redshift assigned.
    """
    if lens_cat is not None:
        z_random = np.random.choice(lens_cat[z_col], len(random_cat))
    else:
        z_random = np.random.uniform(low=z_min, high=z_max, size=len(random_cat))

    if 'z' in random_cat.dtype.names:
        random_cat['z'] = z_random
    else:
        random_cat = append_fields(random_cat, 'z', z_random, usemask=False)

    return random_cat


def corr2cov(arr, corr):
    """Convert a correlation matrix into covariance matrix.

    Parameters
    ----------
    arr : 2-D numpy array
        Array used for correlation matrix.
    corr : 2-D numpy array
        Correlation matrix.

    Return
    ------
        Covariance matrix.
    """
    stddev = np.nanstd(arr, axis=0, ddof=1)

    return stddev[:, None] * corr * stddev[None, :]


def smooth_cov(arr, boxsize=1, trunc=0.2):
    """Estimate a smoothed version of the covariance matrix.

    Parameters
    ----------
    arr : 2-D numpy array

    Return
    ------
    cor_ori : 2-D numpy array
        Original correlation matrix.
    cor_trunc : 2-D numpy array
        Correlation matrix after the smooth and truncation.
    cov_trunc : 2-D numpy array
        Covariance matrix after the smooth and truncation.
    """
    cor_ori = np.corrcoef(np.asarray(arr), rowvar=False)

    # A uniform (boxcar) filter with a width of 1 radial bin
    cor_smooth = ndimage.uniform_filter1d(cor_ori, boxsize, mode='nearest')

    # Truncate the correlation matrix
    cor_trunc = copy.deepcopy(cor_smooth)
    cor_trunc[cor_trunc < trunc] = 0.0

    # Convert the correlation matrix to covariance matrix
    cov_trunc = corr2cov(arr, cor_trunc)

    return cor_ori, cor_trunc, cov_trunc


def _constant_offset(x, a):
    """For estimating the average ratio between two DeltaSigma profile."""
    return x * 0.0 + a


def fit_avg_ratio(rad, dsig_ratio, err, rmin=0.1, rmax=10.0):
    """Estimate the average ratio between two DeltaSigma profiles.

    Parameters
    ----------
    rad : numpy array
        Array for radial bins.
    dsig_ratio : numpy array
        Array for the ratio between two DeltaSigma profiles
    err : numpy array
        1-D errors or 2-D covariance matrix for the ratio.
    rmin : float, optional
        Inner radius boundary. Default: 0.1
    rmax : float, optional
        Outer radius boundary. Default: 10.0

    Returns
    -------
    f : float
        Average ratio between two profiles.
    f_err : float
        Error of the average ratio.

    """
    r_mask = ((rad > rmin) & (rad <= rmax))

    # If error is the covariance matrix, need to cut the matrix
    if err.ndim == 1:
        # This is a 1-D error
        err_use = err[r_mask]
    elif err.ndim == 2:
        # This is a covariance matrix
        idx_min, idx_max = np.argmax(rad > rmin), np.argmin(rad <= rmax)
        err_use = err[idx_min:idx_max, idx_min:idx_max]

    # Fit an average difference
    f, f_cov = curve_fit(_constant_offset, rad[r_mask], dsig_ratio[r_mask],
                         sigma=err_use, p0=1.0)

    return f[0], np.sqrt(f_cov[0][0])
