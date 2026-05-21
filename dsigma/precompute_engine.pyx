# cython: language_level=3, boundscheck=False, wraparound=False
# cython: nonecheck=False, cdivision=True, initializedcheck=False
"""Main computational loop for dsigma."""

import queue as Queue
from libc.math cimport sin, cos, sqrt, fmax, pow

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy_healpix import HEALPix
from scipy.spatial import cKDTree
from tqdm import tqdm


cdef double sigma_crit_factor = (
    1e-6 * c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.pc).value
cdef double dx, dy, dz

cdef double _chord_sq(double sin_ra_1, double cos_ra_1, double sin_dec_1,
                      double cos_dec_1, double sin_ra_2, double cos_ra_2,
                      double sin_dec_2, double cos_dec_2) noexcept:
    """Calculate the chord distance between two points on a unit sphere."""

    dx = cos_ra_1 * cos_dec_1 - cos_ra_2 * cos_dec_2
    dy = sin_ra_1 * cos_dec_1 - sin_ra_2 * cos_dec_2
    dz = sin_dec_1 - sin_dec_2

    return dx * dx + dy * dy + dz * dz


def precompute_engine(
        pix_l, n_pix_l, pix_s, n_pix_s, chord_sq_bins, table_l, table_s,
        table_r, bint comoving, float weighting, int nside, queue,
        progress_bar=False):
    """Sum over all lens-source pairs.

    Parameters
    ----------
    pix_l : numpy.ndarray
        HEALPix pixels containing lenses.
    n_pix_l : numpy.ndarray
        Number of lenses in each HEALPix pixel.
    pix_l : numpy.ndarray
        HEALPix pixels containing sources.
    n_pix_l : numpy.ndarray
        Number of sources in each HEALPix pixel.
    chord_sq_bins : numpy.ndarray
        Angular bins for all lenses defined via chord distances. Has length
        n_l times n_bins, where n_l and n_bins are the number of lenses and
        angular bins edges, respectively.
    table_l : dict
        Table containing information about lenses.
    table_s : dict
        Table containing information about sources.
    table_r : dict
        Table storing results for lenses.
    comoving : bool
        Whether the critical surface density assumes comoving coordinates.
    weighting : float
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
    nside : int
        nside parameter used by HEALPix.
    queue : queue.Queue or multiprocessing.Queue
        Queue tracking which lens pixels remain to be processed.
    progress_bar : bool, optional
        Whether to show a progress bar tracking the processed lens pixels.

    """
    cdef long[::1] n_pix_l_c = n_pix_l
    cdef long[::1] n_pix_s_c = n_pix_s

    cdef double[::1] z_l = table_l['z']
    cdef double[::1] z_s = table_s['z']
    cdef double[::1] d_com_l = table_l['d_com']
    cdef double[::1] d_com_s
    if 'd_com' in table_s:
        d_com_s = table_s['d_com']
    cdef double[::1] sin_ra_l = table_l['sin ra']
    cdef double[::1] cos_ra_l = table_l['cos ra']
    cdef double[::1] sin_dec_l = table_l['sin dec']
    cdef double[::1] cos_dec_l = table_l['cos dec']
    cdef double[::1] sin_ra_s = table_s['sin ra']
    cdef double[::1] cos_ra_s = table_s['cos ra']
    cdef double[::1] sin_dec_s = table_s['sin dec']
    cdef double[::1] cos_dec_s = table_s['cos dec']
    cdef double[::1] w_s = table_s['w']
    cdef double[::1] e_1 = table_s['e_1']
    cdef double[::1] e_2 = table_s['e_2']
    cdef double[::1] z_l_max = table_s['z_l_max']

    cdef bint has_sigma_crit_eff = 'sigma_crit_eff' in table_l
    cdef int n_z_bins = 0
    cdef double[::1] sigma_crit_eff
    cdef long[::1] z_bin
    if has_sigma_crit_eff:
        n_z_bins = len(table_l['sigma_crit_eff']) // len(table_l['z'])
        sigma_crit_eff = table_l['sigma_crit_eff']
        z_bin = table_s['z_bin']

    cdef bint has_m = 'm' in table_s
    cdef double[::1] m
    if has_m:
        m = table_s['m']

    cdef bint has_e_rms = 'e_rms' in table_s
    cdef double[::1] e_rms
    if has_e_rms:
        e_rms = table_s['e_rms']

    cdef bint has_m_sel = 'm_sel' in table_s
    cdef double[::1] m_sel
    if has_m_sel:
        m_sel = table_s['m_sel']

    cdef bint has_R_matrix = 'R_11' in table_s
    cdef double[::1] R_11, R_12, R_21, R_22
    if has_R_matrix:
        R_11 = table_s['R_11']
        R_12 = table_s['R_12']
        R_21 = table_s['R_21']
        R_22 = table_s['R_22']

    cdef double[::1] chord_sq_bins_c = chord_sq_bins

    cdef long[::1] sum_1 = table_r['sum 1']
    cdef double[::1] sum_w_ls = table_r['sum w_ls']
    cdef double[::1] sum_w_ls_e_t = table_r['sum w_ls e_t']
    cdef double[::1] sum_w_ls_e_t_sigma_crit = table_r['sum w_ls e_t sigma_crit']
    cdef double[::1] sum_w_ls_sigma_crit = table_r['sum w_ls sigma_crit']
    cdef double[::1] sum_w_ls_z_s = table_r['sum w_ls z_s']
    cdef double[::1] sum_w_ls_m
    if has_m:
        sum_w_ls_m = table_r['sum w_ls m']
    cdef double[::1] sum_w_ls_1_minus_e_rms_sq
    if has_e_rms:
        sum_w_ls_1_minus_e_rms_sq = table_r['sum w_ls (1 - e_rms^2)']
    cdef double[::1] sum_w_ls_m_sel
    if has_m_sel:
        sum_w_ls_m_sel = table_r['sum w_ls m_sel']
    cdef double[::1] sum_w_ls_R_T
    if has_R_matrix:
        sum_w_ls_R_T = table_r['sum w_ls R_T']

    hp = HEALPix(nside, order='ring')
    lon_pix, lat_pix = hp.healpix_to_lonlat(np.arange(hp.npix))
    x_pix = np.cos(lon_pix) * np.cos(lat_pix)
    y_pix = np.sin(lon_pix) * np.cos(lat_pix)
    z_pix = np.sin(lat_pix)
    xyz_pix = np.array([x_pix, y_pix, z_pix]).T
    xyz_pix_l = xyz_pix[pix_l]
    xyz_pix_s = xyz_pix[pix_s]
    kdtree = cKDTree(xyz_pix_s)
    cdef double[::1] x_pix_l = np.ascontiguousarray(xyz_pix_l[:, 0])
    cdef double[::1] y_pix_l = np.ascontiguousarray(xyz_pix_l[:, 1])
    cdef double[::1] z_pix_l = np.ascontiguousarray(xyz_pix_l[:, 2])
    cdef double[::1] x_pix_s = np.ascontiguousarray(xyz_pix_s[:, 0])
    cdef double[::1] y_pix_s = np.ascontiguousarray(xyz_pix_s[:, 1])
    cdef double[::1] z_pix_s = np.ascontiguousarray(xyz_pix_s[:, 2])

    cdef long i_pix_l, i_l, i_l_min, i_l_max
    cdef long i_pix_s, i_s, i_s_min, i_s_max
    cdef long i_bin, n_bins = len(chord_sq_bins) // len(table_l['z'])  - 1
    cdef long offset_bin, offset_result
    cdef double chord_sq_pix_max, chord_sq_pix_min, chord_sq_ls
    cdef double sin_ra_l_minus_ra_s, cos_ra_l_minus_ra_s
    cdef double sin_2phi, cos_2phi, tan_phi, tan_phi_num, tan_phi_den, e_t
    cdef double w_ls, sigma_crit
    cdef double inf = float('inf'), summand

    # Approximate healpy.pixelfunc.max_pixrad. The following is a
    # reverse-engineered, empirical upper bound.
    cdef double max_pixrad = 1.05 * hp.pixel_resolution.to(u.rad).value

    if progress_bar:
        pbar = tqdm(total=len(pix_l))

    while True:

        # Check whether there is still a lens pixel in the queue.
        try:
            i_pix_l = queue.get(timeout=0.5)
        except Queue.Empty:
            break

        if i_pix_l == 0:
            i_l_min = 0
        else:
            i_l_min = n_pix_l_c[i_pix_l - 1]
        i_l_max = n_pix_l_c[i_pix_l]

        # Find the maximum angular search radius for other pixels.
        chord_sq_pix_max = 0.0
        for i_l in range(i_l_min, i_l_max):
            chord_sq_pix_max = fmax(
                chord_sq_bins_c[i_l * (n_bins + 1) + n_bins], chord_sq_pix_max)

        # Note that galaxies can be up to max_pixrad away from the pixel center.
        chord_sq_pix_max = pow(sqrt(chord_sq_pix_max) + 2 * max_pixrad, 2)

        # Loop over all suitable source pixels.
        for i_pix_s in kdtree.query_ball_point(
                xyz_pix_l[i_pix_l], sqrt(chord_sq_pix_max)):

            if i_pix_s == 0:
                i_s_min = 0
            else:
                i_s_min = n_pix_s_c[i_pix_s - 1]
            i_s_max = n_pix_s_c[i_pix_s]

            # Determine the minimum chord distance of a lens-source pair for
            # this pair of pixels.
            dx = x_pix_l[i_pix_l] - x_pix_s[i_pix_s]
            dy = y_pix_l[i_pix_l] - y_pix_s[i_pix_s]
            dz = z_pix_l[i_pix_l] - z_pix_s[i_pix_s]
            chord_sq_pix_min = dx * dx + dy * dy + dz * dz
            chord_sq_pix_min = pow(sqrt(chord_sq_pix_min) - 2 * max_pixrad, 2)

            # Loop over all lenses in the pixel.
            for i_l in range(i_l_min, i_l_max):

                offset_result = i_l * n_bins
                offset_bin = i_l * (n_bins + 1)

                # Skip this lens if the search radius cannot yield sources
                # in that pixel.
                if chord_sq_bins_c[offset_bin + n_bins] < chord_sq_pix_min:
                    continue

                # Loop over all sources in the pixel.
                for i_s in range(i_s_min, i_s_max):

                    if z_l[i_l] > z_l_max[i_s]:
                        continue

                    chord_sq_ls = _chord_sq(
                        sin_ra_l[i_l], cos_ra_l[i_l], sin_dec_l[i_l],
                        cos_dec_l[i_l], sin_ra_s[i_s], cos_ra_s[i_s],
                        sin_dec_s[i_s], cos_dec_s[i_s])

                    i_bin = n_bins
                    while i_bin >= 0:
                        if chord_sq_ls > chord_sq_bins_c[offset_bin + i_bin]:
                            break
                        i_bin -= 1

                    if i_bin == n_bins or i_bin < 0:
                        continue

                    if w_s[i_s] == 0:
                        continue

                    if has_sigma_crit_eff:
                        sigma_crit = sigma_crit_eff[
                            i_l * n_z_bins + z_bin[i_s]]
                    elif d_com_l[i_l] < d_com_s[i_s]:
                        sigma_crit = (sigma_crit_factor * (1 + z_l[i_l]) *
                            d_com_s[i_s] / d_com_l[i_l] /
                            (d_com_s[i_s] - d_com_l[i_l]))
                        if comoving:
                            sigma_crit /= (1.0 + z_l[i_l]) * (1.0 + z_l[i_l])
                    else:
                        if weighting < 0:
                            continue
                        else:
                            sigma_crit = inf

                    if weighting == 0:
                        w_ls = w_s[i_s]
                    elif weighting == -2:
                         w_ls = w_s[i_s] / sigma_crit / sigma_crit
                    else:
                        w_ls = w_s[i_s] * pow(sigma_crit, weighting)

                    if w_ls == 0:
                        continue

                    sin_ra_l_minus_ra_s = (sin_ra_l[i_l] * cos_ra_s[i_s] -
                                           cos_ra_l[i_l] * sin_ra_s[i_s])
                    cos_ra_l_minus_ra_s = (cos_ra_l[i_l] * cos_ra_s[i_s] +
                                           sin_ra_l[i_l] * sin_ra_s[i_s])
                    tan_phi_num = (cos_dec_s[i_s] * sin_dec_l[i_l] - sin_dec_s[i_s] *
                                   cos_dec_l[i_l] * cos_ra_l_minus_ra_s)
                    tan_phi_den = cos_dec_l[i_l] * sin_ra_l_minus_ra_s
                    if tan_phi_den == 0:
                        cos_2phi = -1
                        sin_2phi = 0
                    else:
                        tan_phi = tan_phi_num / tan_phi_den
                        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
                        sin_2phi = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi)

                    e_t = - e_1[i_s] * cos_2phi + e_2[i_s] * sin_2phi

                    sum_1[offset_result + i_bin] += 1
                    summand = w_ls
                    sum_w_ls[offset_result + i_bin] += summand
                    summand *= e_t
                    sum_w_ls_e_t[offset_result + i_bin] += summand
                    summand *= sigma_crit
                    sum_w_ls_e_t_sigma_crit[offset_result + i_bin] += summand
                    sum_w_ls_z_s[offset_result + i_bin] += w_ls * z_s[i_s]
                    sum_w_ls_sigma_crit[offset_result + i_bin] += w_ls * sigma_crit
                    if has_m:
                        sum_w_ls_m[offset_result + i_bin] += w_ls * m[i_s]
                    if has_e_rms:
                        sum_w_ls_1_minus_e_rms_sq[offset_result + i_bin] += (
                            w_ls * (1 - e_rms[i_s] * e_rms[i_s]))
                    if has_m_sel:
                        sum_w_ls_m_sel[offset_result + i_bin] += w_ls * m_sel[i_s]
                    if has_R_matrix:
                        sum_w_ls_R_T[offset_result + i_bin] += w_ls * (
                            R_11[i_s] * cos_2phi * cos_2phi +
                            R_22[i_s] * sin_2phi * sin_2phi +
                            (R_12[i_s] + R_21[i_s]) * sin_2phi * cos_2phi)

        if progress_bar:
            pbar.update(i_pix_l + 1 - pbar.n)

    if progress_bar:
        pbar.update(len(pix_l) - pbar.n)
        pbar.close()
