# cython: language_level=3, boundscheck=False, wraparound=False
# cython: nonecheck=False, cdivision=True, initializedcheck=False

import queue as Queue
import numpy as np
import healpy as hp
from libc.math cimport sin, cos, sqrt, fmax
from scipy.spatial import cKDTree

from astropy import constants as c
from astropy import units as u


cdef double sigma_crit_factor = (
    1e-6 * c.c**2 / (4 * np.pi * c.G)).to(u.Msun / u.pc).value
cdef double deg2rad = np.pi / 180.0

cdef double x_1, x_2, y_1, y_2, z_1, z_2

cdef dist_3d_sq(double sin_ra_1, double cos_ra_1, double sin_dec_1,
                double cos_dec_1, double sin_ra_2, double cos_ra_2,
                double sin_dec_2, double cos_dec_2):

    x_1 = cos_ra_1 * cos_dec_1
    y_1 = sin_ra_1 * cos_dec_1
    z_1 = sin_dec_1
    x_2 = cos_ra_2 * cos_dec_2
    y_2 = sin_ra_2 * cos_dec_2
    z_2 = sin_dec_2

    return ((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2) +
            (z_1 - z_2) * (z_1 - z_2))


def precompute_engine(
        pix_l_counts_in, pix_s_counts_in, pix_l_cum_counts_in,
        pix_s_cum_counts_in, z_l_in, z_s_in, d_com_l_in, d_com_s_in,
        sin_ra_l_in, cos_ra_l_in, sin_dec_l_in, cos_dec_l_in, sin_ra_s_in,
        cos_ra_s_in, sin_dec_s_in, cos_dec_s_in, w_s_in, e_1_in, e_2_in,
        z_l_max_in, f_bias_in, z_bin_in, sigma_crit_eff_in, m_in, e_rms_in,
        R_2_in, R_11_in, R_12_in, R_21_in, R_22_in, dist_3d_sq_bins_in,
        sum_1_in, sum_w_ls_in, sum_w_ls_e_t_in, sum_w_ls_e_t_sigma_crit_in,
        sum_w_ls_e_t_sigma_crit_f_bias_in, sum_w_ls_e_t_sigma_crit_sq_in,
        sum_w_ls_z_s_in, sum_w_ls_m_in, sum_w_ls_1_minus_e_rms_sq_in,
        sum_w_ls_A_p_R_2_in, sum_w_ls_R_T_in, bins, bint comoving,
        bint shear_mode, int nside, queue):

    cdef long[::1] pix_l_counts = pix_l_counts_in
    cdef long[::1] pix_s_counts = pix_s_counts_in
    cdef long[::1] pix_l_cum_counts = pix_l_cum_counts_in
    cdef long[::1] pix_s_cum_counts = pix_s_cum_counts_in
    cdef double[::1] z_l = z_l_in
    cdef double[::1] z_s = z_s_in
    cdef double[::1] d_com_l = d_com_l_in
    cdef double[::1] d_com_s = d_com_s_in
    cdef double[::1] sin_ra_l = sin_ra_l_in
    cdef double[::1] cos_ra_l = cos_ra_l_in
    cdef double[::1] sin_dec_l = sin_dec_l_in
    cdef double[::1] cos_dec_l = cos_dec_l_in
    cdef double[::1] sin_ra_s = sin_ra_s_in
    cdef double[::1] cos_ra_s = cos_ra_s_in
    cdef double[::1] sin_dec_s = sin_dec_s_in
    cdef double[::1] cos_dec_s = cos_dec_s_in
    cdef double[::1] w_s = w_s_in
    cdef double[::1] e_1 = e_1_in
    cdef double[::1] e_2 = e_2_in
    cdef double[::1] z_l_max = z_l_max_in

    cdef bint has_f_bias = f_bias_in is not None
    cdef double[::1] f_bias
    if has_f_bias:
        f_bias = f_bias_in

    cdef bint has_sigma_crit_eff = sigma_crit_eff_in is not None
    cdef int n_z_bins = 0
    cdef double[::1] sigma_crit_eff
    cdef long[::1] z_bin
    if has_sigma_crit_eff:
        n_z_bins = len(sigma_crit_eff_in) // len(z_l_in)
        sigma_crit_eff = sigma_crit_eff_in
        z_bin = z_bin_in

    cdef bint has_m = m_in is not None
    cdef double[::1] m
    if has_m:
        m = m_in

    cdef bint has_e_rms = e_rms_in is not None
    cdef double[::1] e_rms
    if has_e_rms:
        e_rms = e_rms_in

    cdef bint has_R_2 = R_2_in is not None
    cdef double[::1] R_2
    if has_R_2:
        R_2 = R_2_in

    cdef bint has_R_matrix = R_11_in is not None
    cdef double[::1] R_11, R_12, R_21, R_22
    if has_R_matrix:
        R_11 = R_11_in
        R_12 = R_12_in
        R_21 = R_21_in
        R_22 = R_22_in

    cdef double[::1] dist_3d_sq_bins = dist_3d_sq_bins_in

    cdef long[::1] sum_1 = sum_1_in
    cdef double[::1] sum_w_ls = sum_w_ls_in
    cdef double[::1] sum_w_ls_e_t = sum_w_ls_e_t_in
    cdef double[::1] sum_w_ls_e_t_sigma_crit = sum_w_ls_e_t_sigma_crit_in
    cdef double[::1] sum_w_ls_e_t_sigma_crit_f_bias
    if has_f_bias:
        sum_w_ls_e_t_sigma_crit_f_bias = sum_w_ls_e_t_sigma_crit_f_bias_in
    cdef double[::1] sum_w_ls_e_t_sigma_crit_sq = sum_w_ls_e_t_sigma_crit_sq_in
    cdef double[::1] sum_w_ls_z_s = sum_w_ls_z_s_in
    cdef double[::1] sum_w_ls_m
    if has_m:
        sum_w_ls_m = sum_w_ls_m_in
    cdef double[::1] sum_w_ls_1_minus_e_rms_sq
    if has_e_rms:
        sum_w_ls_1_minus_e_rms_sq = sum_w_ls_1_minus_e_rms_sq_in
    cdef double[::1] sum_w_ls_A_p_R_2
    if has_R_2:
        sum_w_ls_A_p_R_2 = sum_w_ls_A_p_R_2_in
    cdef double[::1] sum_w_ls_R_T
    if has_R_matrix:
        sum_w_ls_R_T = sum_w_ls_R_T_in

    x, y, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    xyz = np.array([x, y, z]).T
    kdtree = cKDTree(xyz)

    cdef long i_l, i_l_min, i_l_max
    cdef long i_s, i_s_min, i_s_max
    cdef long i_pix_l, i_pix_s
    cdef long[::1] i_pix_s_list
    cdef long i_bin, n_bins = len(bins) - 1
    cdef long offset_bin, offset_result
    cdef double dist_3d_sq_max, dist_3d_sq_ls
    cdef double sin_ra_l_minus_ra_s, cos_ra_l_minus_ra_s
    cdef double sin_2phi, cos_2phi, tan_phi, e_t
    cdef double w_ls, sigma_crit
    cdef double max_pixrad = hp.max_pixrad(nside, degrees=True)

    while True:

        # Check whether there is still a lens pixel in the queue.
        try:
            i_pix_l = queue.get(timeout=0.5)
        except Queue.Empty:
            break

        # Go to next pixel if current pixel does not contain any lenses.
        if pix_l_counts[i_pix_l] == 0:
            continue

        if i_pix_l == 0:
            i_l_min = 0
        else:
            i_l_min = pix_l_cum_counts[i_pix_l - 1]
        i_l_max = pix_l_cum_counts[i_pix_l]

        # Find the maximum angular search radius.
        dist_3d_sq_max = 0.0
        for i_l in range(i_l_min, i_l_max):
            dist_3d_sq_max = fmax(dist_3d_sq_bins[i_l * (n_bins + 1) + n_bins],
                                  dist_3d_sq_max)
        # Note that pixels can be up to max_pixrad away from the pixel center.
        dist_3d_sq_max += (4 * deg2rad * deg2rad * max_pixrad * max_pixrad +
                           4 * sqrt(dist_3d_sq_max) * deg2rad * max_pixrad)

        # Get list of all source pixels that could contain suitable sources.
        i_pix_s_list = np.fromiter(
            kdtree.query_ball_point(xyz[i_pix_l], sqrt(dist_3d_sq_max)),
            dtype=long)

        # Loop over all suitable source pixels.
        for i_pix_s in i_pix_s_list:

            # Go to next pixel if current pixel does not contain any sources.
            if pix_s_counts[i_pix_s] == 0:
                continue

            if i_pix_s == 0:
                i_s_min = 0
            else:
                i_s_min = pix_s_cum_counts[i_pix_s - 1]
            i_s_max = pix_s_cum_counts[i_pix_s]

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

                    if dist_3d_sq_ls > dist_3d_sq_bins[offset_bin + n_bins]:
                        continue

                    if shear_mode:
                        w_ls = 1
                        sigma_crit = 1
                    elif has_sigma_crit_eff:
                        sigma_crit = sigma_crit_eff[
                            i_l * n_z_bins + z_bin[i_s]]
                        w_ls = w_s[i_s] / sigma_crit / sigma_crit
                    elif z_l[i_l] < z_s[i_s]:
                        sigma_crit = (sigma_crit_factor * (1 + z_l[i_l]) *
                            d_com_s[i_s] / d_com_l[i_l] /
                            (d_com_s[i_s] - d_com_l[i_l]))
                        if comoving:
                            sigma_crit /= (1.0 + z_l[i_l]) * (1.0 + z_l[i_l])
                        w_ls = w_s[i_s] / sigma_crit / sigma_crit
                    else:
                        sigma_crit = 0
                        w_ls = 0

                    sin_ra_l_minus_ra_s = (sin_ra_l[i_l] * cos_ra_s[i_s] -
                                           cos_ra_l[i_l] * sin_ra_s[i_s])
                    cos_ra_l_minus_ra_s = (cos_ra_l[i_l] * cos_ra_s[i_s] +
                                           sin_ra_l[i_l] * sin_ra_s[i_s])

                    if cos_dec_l[i_l] * sin_ra_l_minus_ra_s == 0:

                        cos_2phi = -1
                        sin_2phi = 0

                    else:

                        tan_phi = (
                            (cos_dec_s[i_s] * sin_dec_l[i_l] - sin_dec_s[i_s] *
                             cos_dec_l[i_l] * cos_ra_l_minus_ra_s) /
                            (cos_dec_l[i_l] * sin_ra_l_minus_ra_s))

                        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
                        sin_2phi = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi)

                    e_t = - e_1[i_s] * cos_2phi + e_2[i_s] * sin_2phi

                    # Loop over bins going from the outermost inwards.
                    i_bin = n_bins - 1
                    while i_bin >= 0:

                        if dist_3d_sq_ls > dist_3d_sq_bins[offset_bin + i_bin]:
                            sum_1[offset_result + i_bin] += 1
                            sum_w_ls[offset_result + i_bin] += w_ls
                            sum_w_ls_e_t[offset_result + i_bin] += w_ls * e_t
                            if not shear_mode:
                                sum_w_ls_e_t_sigma_crit[offset_result + i_bin] += (
                                    w_ls * e_t * sigma_crit)
                                if has_f_bias:
                                    sum_w_ls_e_t_sigma_crit_f_bias[offset_result + i_bin] += (
                                        w_ls * e_t * sigma_crit * f_bias[i_l])
                                sum_w_ls_e_t_sigma_crit_sq[offset_result + i_bin] += (
                                    w_ls * e_t * sigma_crit)**2
                            sum_w_ls_z_s[offset_result + i_bin] += w_ls * z_s[i_s]
                            if has_m:
                                sum_w_ls_m[offset_result + i_bin] += w_ls * m[i_s]
                            if has_e_rms:
                                sum_w_ls_1_minus_e_rms_sq[offset_result + i_bin] += (
                                    w_ls * (1 - e_rms[i_s]**2))
                            if has_R_2 and R_2[i_s] <= 0.31:
                                sum_w_ls_A_p_R_2[offset_result + i_bin] += (
                                    0.00865 * w_ls / 0.01)
                            if has_R_matrix:
                                sum_w_ls_R_T[offset_result + i_bin] += w_ls * (
                                    R_11[i_s] * cos_2phi**2 +
                                    R_22[i_s] * sin_2phi**2 +
                                    (R_12[i_s] + R_21[i_s]) * sin_2phi *
                                    cos_2phi)
                            break

                        i_bin -= 1

    return 0
