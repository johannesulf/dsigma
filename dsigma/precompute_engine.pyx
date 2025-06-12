# cython: language_level=3, boundscheck=False, wraparound=False
# cython: nonecheck=False, cdivision=True, initializedcheck=False

import queue as Queue
import numpy as np
from tqdm import tqdm
from astropy_healpix import HEALPix
from astropy import units as u
from libc.math cimport sin, cos, sqrt, fmax, pow
from scipy.spatial import cKDTree

from astropy import constants as c
from astropy import units as u


cdef double sigma_crit_factor = (
    1e-6 * c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.pc).value
cdef double deg2rad = np.pi / 180.0

cdef double dx, dy, dz

cdef double dist_3d_sq(double sin_ra_1, double cos_ra_1, double sin_dec_1,
                       double cos_dec_1, double sin_ra_2, double cos_ra_2,
                       double sin_dec_2, double cos_dec_2):

    dx = cos_ra_1 * cos_dec_1 - cos_ra_2 * cos_dec_2
    dy = sin_ra_1 * cos_dec_1 - sin_ra_2 * cos_dec_2
    dz = sin_dec_1 - sin_dec_2

    return dx * dx + dy * dy + dz * dz


def precompute_engine(
        u_pix_l, n_pix_l_in, u_pix_s, n_pix_s_in, dist_3d_sq_bins_in,
        table_l, table_s, table_r, bins, bint comoving, float weighting,
        int nside, queue, progress_bar):

    cdef long[::1] n_pix_l = n_pix_l_in
    cdef long[::1] n_pix_s = n_pix_s_in

    cdef double[::1] z_l = table_l['z']
    cdef double[::1] z_s = table_s['z']
    cdef double[::1] d_com_l = table_l['d_com']
    cdef double[::1] d_com_s = table_s['d_com']
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

    cdef bint has_sigma_crit_eff = 'sigma_crit_eff' in table_l.keys()
    cdef int n_z_bins = 0
    cdef double[::1] sigma_crit_eff
    cdef long[::1] z_bin
    if has_sigma_crit_eff:
        n_z_bins = len(table_l['sigma_crit_eff']) // len(table_l['z'])
        sigma_crit_eff = table_l['sigma_crit_eff']
        z_bin = table_s['z_bin']

    cdef bint has_m = 'm' in table_s.keys()
    cdef double[::1] m
    if has_m:
        m = table_s['m']

    cdef bint has_e_rms = 'e_rms' in table_s.keys()
    cdef double[::1] e_rms
    if has_e_rms:
        e_rms = table_s['e_rms']

    cdef bint has_m_sel = 'm_sel' in table_s.keys()
    cdef double[::1] m_sel
    if has_m_sel:
        m_sel = table_s['m_sel']

    cdef bint has_R_matrix = 'R_11' in table_s.keys()
    cdef double[::1] R_11, R_12, R_21, R_22
    if has_R_matrix:
        R_11 = table_s['R_11']
        R_12 = table_s['R_12']
        R_21 = table_s['R_21']
        R_22 = table_s['R_22']

    cdef double[::1] dist_3d_sq_bins = dist_3d_sq_bins_in

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
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    xyz = np.array([x, y, z]).T
    xyz_l = xyz[u_pix_l]
    xyz_s = xyz[u_pix_s]
    kdtree = cKDTree(xyz_s)

    cdef long pix_l, i_l, i_l_min, i_l_max
    cdef long pix_s, i_pix_s, l_pix_s, i_s, i_s_min, i_s_max
    cdef long[::1] pix_s_list
    cdef long i_bin, n_bins = len(bins) - 1
    cdef long offset_bin, offset_result
    cdef double dist_3d_sq_max, dist_3d_sq_ls
    cdef double sin_ra_l_minus_ra_s, cos_ra_l_minus_ra_s
    cdef double sin_2phi, cos_2phi, tan_phi, tan_phi_num, tan_phi_den, e_t
    cdef double w_ls, sigma_crit
    cdef double max_pixrad = 1.05 * hp.pixel_resolution.to(u.deg).value
    cdef double inf = float('inf'), summand

    if progress_bar:
        pbar = tqdm(total=len(u_pix_l))

    while True:

        # Check whether there is still a lens pixel in the queue.
        try:
            pix_l = queue.get(timeout=0.5)
        except Queue.Empty:
            break

        if pix_l == 0:
            i_l_min = 0
        else:
            i_l_min = n_pix_l[pix_l - 1]
        i_l_max = n_pix_l[pix_l]

        # Find the maximum angular search radius.
        dist_3d_sq_max = 0.0
        for i_l in range(i_l_min, i_l_max):
            dist_3d_sq_max = fmax(dist_3d_sq_bins[i_l * (n_bins + 1) + n_bins],
                                  dist_3d_sq_max)
        # Note that pixels can be up to max_pixrad away from the pixel center.
        dist_3d_sq_max += (4 * deg2rad * deg2rad * max_pixrad * max_pixrad +
                           4 * sqrt(dist_3d_sq_max) * deg2rad * max_pixrad)

        # Get list of all source pixels that could contain suitable sources.
        pix_s_list = np.fromiter(
            kdtree.query_ball_point(xyz_l[pix_l], sqrt(dist_3d_sq_max)),
            dtype=int)
        l_pix_s = len(pix_s_list)

        # Loop over all suitable source pixels.
        for i_pix_s in range(l_pix_s):

            pix_s = pix_s_list[i_pix_s]
            if pix_s == 0:
                i_s_min = 0
            else:
                i_s_min = n_pix_s[pix_s - 1]
            i_s_max = n_pix_s[pix_s]

            # Loop over all lenses in the pixel.
            for i_l in range(i_l_min, i_l_max):

                offset_result = i_l * n_bins
                offset_bin = i_l * (n_bins + 1)

                # Loop over all sources in the pixel.
                for i_s in range(i_s_min, i_s_max):

                    if z_l[i_l] > z_l_max[i_s]:
                        continue

                    dist_3d_sq_ls = dist_3d_sq(
                        sin_ra_l[i_l], cos_ra_l[i_l], sin_dec_l[i_l],
                        cos_dec_l[i_l], sin_ra_s[i_s], cos_ra_s[i_s],
                        sin_dec_s[i_s], cos_dec_s[i_s])

                    i_bin = n_bins
                    while i_bin >= 0:
                        if dist_3d_sq_ls > dist_3d_sq_bins[offset_bin + i_bin]:
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
            pbar.update(pix_l + 1 - pbar.n)

    if progress_bar:
        pbar.update(len(u_pix_l) - pbar.n)
        pbar.close()

    return 0
