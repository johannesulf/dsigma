"""QA plots."""
from __future__ import division, print_function

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt

rc('text', usetex=True)

__all__ = ["plot_r_dsigma_tx", "plot_r_npairs", "plot_r_boost_factor",
           "plot_r_calibration_factors", "plot_r_delta_sigma",
           "plot_delta_sigma", "qa_rand_photoz_weight", "plot_corr_matrix",
           "qa_lens_rand_zhist", "show_delta_sigma_profiles", 'show_single_profile']

np.seterr(divide='ignore', invalid='ignore')

Dark2 = plt.get_cmap('Dark2')


def plot_r_dsigma_tx(dsigma_output, random=False, qa_title=r'$\Delta\Sigma\ \mathrm{Profile}$',
                     qa_prefix='qa_r_dsigma_tx'):
    """Check all delta sigma signals."""
    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.99,
                        bottom=0.14, top=0.93,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.set_title(qa_title, fontsize=30)
    ax1.set_xscale("log", nonposx='clip')

    r_mpc = dsigma_output['r_mpc']
    lens_dsigma_t = dsigma_output['lens_dsigma_t']
    lens_dsigma_x = dsigma_output['lens_dsigma_x']
    lens_calib = dsigma_output['lens_calib']
    if random:
        rand_dsigma_t = dsigma_output['rand_dsigma_t']
        rand_dsigma_x = dsigma_output['rand_dsigma_x']
        rand_calib = dsigma_output['rand_calib']

    ax1.plot(r_mpc, lens_dsigma_t, c=np.asarray(Dark2(0.1)),
             label=r'$\mathrm{Lens:\ tan}$', linewidth=2)
    ax1.plot(r_mpc, lens_dsigma_x, c=np.asarray(Dark2(0.1)),
             label=r'$\mathrm{Lens:\ crx}$', linewidth=2, linestyle='--')

    ax1.plot(r_mpc, lens_dsigma_t * lens_calib, c=np.asarray(Dark2(0.6)),
             label=r'$\mathrm{Lens:\ tan, calib}$', linewidth=2)
    ax1.plot(r_mpc, lens_dsigma_x * lens_calib, c=np.asarray(Dark2(0.6)),
             label=r'$\mathrm{Lens:\ crx, calib}$', linewidth=2, linestyle='--')

    if random:
        ax1.plot(r_mpc, rand_dsigma_t, c=np.asarray(Dark2(0.2)),
                 label=r'$\mathrm{Rand:\ tan}$', linewidth=2)
        ax1.plot(r_mpc, rand_dsigma_x, c=np.asarray(Dark2(0.2)),
                 label=r'$\mathrm{Rand:\ crx}$', linewidth=2, linestyle='--')

        ax1.plot(r_mpc, rand_dsigma_t * rand_calib, c=np.asarray(Dark2(0.5)),
                 label=r'$\mathrm{Rand:\ tan, calib}$', linewidth=2)
        ax1.plot(r_mpc, rand_dsigma_x * rand_calib, c=np.asarray(Dark2(0.5)),
                 label=r'$\mathrm{Rand:\ crx, calib}$', linewidth=2, linestyle='--')

    ax1.legend(fontsize=14, loc="best")

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.set_xlabel(r'$R/\mathrm{Mpc}$', fontsize=25)
    ax1.set_ylabel(r'$\Delta\Sigma\ (M_{\odot}/\mathrm{pc}^2)$',
                   fontsize=25)

    fig.savefig(qa_prefix + '.png', dpi=100)


def plot_r_npairs(dsigma_output, qa_title=r'$N\mathrm{\ Lens-Source\ Pairs}$',
                  qa_prefix='qa_r_npairs', random=False):
    """Plot the profiles of pair number."""
    r_mpc = dsigma_output['r_mpc']
    n_lens = dsigma_output['lens_npairs']

    if random:
        n_rand = dsigma_output['rand_npairs']
        n_rand_eff = dsigma_output['rand_npairs_eff']

    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.99,
                        bottom=0.14, top=0.93,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.set_title(qa_title, fontsize=30)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax1.scatter(r_mpc, n_lens, marker='o', s=35, facecolor=np.asarray(Dark2(0.1)),
                linewidth=2, edgecolor=np.asarray(Dark2(0.2)),
                label=r'$\mathrm{Lens}$')
    if random:
        ax1.scatter(r_mpc, n_rand, marker='h', s=35, facecolor=np.asarray(Dark2(0.5)),
                    linewidth=2, edgecolor=np.asarray(Dark2(0.4)),
                    label=r'$\mathrm{Random}$')

        ax1.scatter(r_mpc, n_rand_eff, marker='h', s=30, c=np.asarray(Dark2(0.3)),
                    label=r'$\mathrm{Random\ Effective}$')

    ax1.legend(fontsize=15, loc='best')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.set_xlabel(r'$R/\mathrm{Mpc}$', fontsize=25)
    ax1.set_ylabel(r'$\mathrm{Number\ of\ Pairs}$', fontsize=25)

    fig.savefig(qa_prefix + '.png', dpi=100)


def plot_r_boost_factor(dsigma_output,
                        qa_title=r'$\mathrm{Boost\ factor}$',
                        qa_prefix='qa_r_boost_factors'):
    """Plot the profiles of boost correction factors."""
    r_mpc, boost = dsigma_output['r_mpc'], dsigma_output['boost_factor']

    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.99,
                        bottom=0.14, top=0.93,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.set_title(qa_title, fontsize=30)
    ax1.set_xscale("log", nonposx='clip')

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax1.scatter(r_mpc, boost, marker='o', s=30, facecolor=np.asarray(Dark2(0.0)),
                edgecolor='k')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.set_xlabel(r'$\log\ (R/\mathrm{Mpc})$', fontsize=25)
    ax1.set_ylabel(r'$\mathrm{Boost\ correction\ factor}$', fontsize=25)

    fig.savefig(qa_prefix + '.png', dpi=100)


def plot_r_calibration_factors(dsigma_output, random=False,
                               qa_title=r'$\mathrm{Calibration\ Factors}$',
                               qa_prefix='qa_r_calibration_factors'):
    """Plot the profiles of calibration factors."""
    r_mpc = dsigma_output['r_mpc']
    lens_r = dsigma_output['lens_r']
    lens_k = dsigma_output['lens_k']
    lens_m = dsigma_output['lens_m']
    lens_c = dsigma_output['lens_calib']

    if random:
        rand_r = dsigma_output['rand_r']
        rand_k = dsigma_output['rand_k']
        rand_m = dsigma_output['rand_m']
        rand_c = dsigma_output['rand_calib']

    fig = plt.figure(figsize=(11, 9))
    fig.suptitle(qa_title, fontsize=25, y=1.002)
    fig.subplots_adjust(left=0.10, right=0.99,
                        bottom=0.08, top=0.95,
                        wspace=0.26, hspace=0.03)

    # Panel 1: 2 \times R
    ax1 = fig.add_subplot(221)
    ax1.set_xscale("log", nonposx='clip')
    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax1.scatter(r_mpc, 2 * lens_r, marker='o',
                facecolor=np.asarray(Dark2(0.1)), edgecolor=np.asarray(Dark2(0.2)),
                s=30, alpha=1.0, label=r'$\mathrm{Lens}$')
    if random:
        ax1.scatter(r_mpc, 2 * rand_r, marker='h', c=np.asarray(Dark2(0.5)),
                    s=25, alpha=0.9, label=r'$\mathrm{Random}$')

    ax1.legend(fontsize=18, loc='best')

    ax1.get_xaxis().set_ticks([])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    ax1.set_ylabel(r'$[2 \times R]\ \mathrm{Factor}$', fontsize=22)

    # Panel 2: 1 + k
    ax2 = fig.add_subplot(222)
    ax2.set_xscale("log", nonposx='clip')
    ax2.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax2.scatter(r_mpc, 1 + lens_k, marker='o',
                facecolor=np.asarray(Dark2(0.1)), edgecolor=np.asarray(Dark2(0.2)),
                s=30, alpha=1.0, label=r'$\mathrm{Lens}$')
    if random:
        ax2.scatter(r_mpc, 1 + rand_k, marker='h', c=np.asarray(Dark2(0.5)),
                    s=25, alpha=0.9, label=r'$\mathrm{Random}$')

    ax2.legend(fontsize=18, loc='best')

    ax2.get_xaxis().set_ticks([])
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax2.set_ylabel(r'$[1 + K]\ \mathrm{Factor}$', fontsize=22)

    # Panel 3: 1 + m
    ax3 = fig.add_subplot(223)
    ax3.set_xscale("log", nonposx='clip')
    ax3.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax3.scatter(r_mpc, 1 + lens_m, marker='o',
                facecolor=np.asarray(Dark2(0.1)), edgecolor=np.asarray(Dark2(0.2)),
                s=30, alpha=1.0, label=r'$\mathrm{Lens}$')
    if random:
        ax3.scatter(r_mpc, 1 + rand_m, marker='h', c=np.asarray(Dark2(0.5)),
                    s=25, alpha=0.9, label=r'$\mathrm{Random}$')

    ax3.legend(fontsize=18, loc='best')

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax3.set_xlabel(r'$R/\mathrm{Mpc}$', fontsize=22)
    ax3.set_ylabel(r'$[1 + m_{\mathrm{sel}}]\ \mathrm{Factor}$', fontsize=22)

    # Panel 4: Calibration factors
    ax4 = fig.add_subplot(224)
    ax4.set_xscale("log", nonposx='clip')
    ax4.grid(linestyle='--', alpha=0.2, linewidth=2)

    ax4.scatter(r_mpc, lens_c, marker='o',
                facecolor=np.asarray(Dark2(0.1)), edgecolor=np.asarray(Dark2(0.2)),
                s=30, alpha=1.0, label=r'$\mathrm{Lens}$')
    if random:
        ax4.scatter(r_mpc, rand_c, marker='h', c=np.asarray(Dark2(0.5)),
                    s=25, alpha=0.9, label=r'$\mathrm{Random}$')

    ax4.legend(fontsize=18, loc='best')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax4.set_xlabel(r'$R/\mathrm{Mpc}$', fontsize=22)
    ax4.set_ylabel(r'$\mathrm{Calibration\ Factor}$', fontsize=22)

    fig.savefig(qa_prefix + '.png', dpi=120)


def plot_r_delta_sigma(dsigma_output,
                       qa_title=r'$R \times \Delta\Sigma\ \mathrm{Profile}$',
                       qa_prefix='qa_r_delta_sigma_profile'):
    """Plot the R v.s. RxDeltaSigma."""
    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.99,
                        bottom=0.14, top=0.93,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.set_title(qa_title, fontsize=30)
    ax1.set_xscale("log", nonposx='clip')

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    r_mpc = dsigma_output['r_mpc']
    delta_sigma = dsigma_output['dsigma_lr']

    # Simple Error
    err_1 = dsigma_output['dsigma_err_1']
    ax1.errorbar(r_mpc * (1.07 ** 0.0), r_mpc * delta_sigma, yerr=(r_mpc * err_1),
                 ecolor=np.asarray(Dark2(0.0)), color=np.asarray(Dark2(0.0)),
                 fmt='o', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.9, label=r'$\mathrm{Simple\ err}$')

    # GGLens Error
    err_2 = dsigma_output['dsigma_err_2']
    ax1.errorbar(r_mpc * (1.07 ** 1.0), r_mpc * delta_sigma, yerr=(r_mpc * err_2),
                 ecolor=np.asarray(Dark2(0.1)), color=np.asarray(Dark2(0.1)),
                 fmt='h', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.9, label=r'$\mathrm{GGlens\ err}$')

    # Jackknife Error
    err_3 = dsigma_output['dsigma_err_jk']
    ax1.errorbar(r_mpc * (1.07 ** 2.0), r_mpc * delta_sigma, yerr=(r_mpc * err_3),
                 ecolor=np.asarray(Dark2(0.2)), color=np.asarray(Dark2(0.2)),
                 fmt='h', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.8, label=r'$\mathrm{Jackknife\ err}$')

    ax1.legend(fontsize=16, loc='upper right')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.set_xlabel(r'$\log\ (R/\mathrm{Mpc})$', fontsize=25)
    yl = (r'$R \times$' +
          r'$\Delta\Sigma\ (\mathrm{Mpc}\ M_{\odot}/\mathrm{pc}^2)$')
    ax1.set_ylabel(yl, fontsize=25)

    fig.savefig(qa_prefix + '.png', dpi=100)


def plot_delta_sigma(dsigma_output, samples=None,
                     qa_title=r'$\Delta\Sigma\ \mathrm{Profile}$',
                     qa_prefix='qa_delta_sigma_profile'):
    """Plot the R v.s. DeltaSigma profile."""
    r_mpc = dsigma_output['r_mpc']
    delta_sigma = dsigma_output['dsigma_lr']

    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.17, right=0.99,
                        bottom=0.14, top=0.93,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)
    ax1.set_title(qa_title, fontsize=30)
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    # Simple error
    err_1 = dsigma_output['dsigma_err_1']
    ax1.errorbar(r_mpc * (1.07 ** 0.0), delta_sigma, yerr=err_1,
                 ecolor=np.asarray(Dark2(0.0)), color=np.asarray(Dark2(0.0)),
                 fmt='o', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.9, label=r'$\mathrm{Simple\ err}$')

    # GGLens Error
    err_2 = dsigma_output['dsigma_err_2']
    ax1.errorbar(r_mpc * (1.07 ** 1.0), delta_sigma, yerr=err_2,
                 ecolor=np.asarray(Dark2(0.1)), color=np.asarray(Dark2(0.1)),
                 fmt='h', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.9, label=r'$\mathrm{GGlens\ err}$')

    # Jackknife Error
    err_3 = dsigma_output['dsigma_err_jk']
    ax1.errorbar(r_mpc * (1.07 ** 2.0), delta_sigma, yerr=err_3,
                 ecolor=np.asarray(Dark2(0.2)), color=np.asarray(Dark2(0.2)),
                 fmt='h', capsize=2, capthick=1.5, elinewidth=1.5,
                 markersize=8, alpha=0.8, label=r'$\mathrm{Jackknife\ err}$')

    ax1.legend(fontsize=16, loc='upper right')

    if samples is not None:
        for prof in samples:
            prof = np.where(prof <= 0, np.nan, prof)
            ax1.scatter(r_mpc, prof, s=15, alpha=0.4, c=np.asarray(Dark2(0.4)),
                        marker='+')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    ax1.set_xlabel(r'$R/\mathrm{Mpc}$', fontsize=25)
    ax1.set_ylabel(r'$\Delta\Sigma\ (M_{\odot}/\mathrm{pc}^2)$',
                   fontsize=25)

    fig.savefig(qa_prefix + '.png', dpi=100)


def qa_rand_photoz_weight(zbin_center, z_weights,
                          qa_title='Re-weight photo-z for random',
                          qa_prefix=None):
    """Plot redshift vs. weight for randoms."""
    if qa_prefix is None:
        qa_prefix = 'qa_rand_photoz_weight'
    else:
        qa_prefix += '_qa_rand_photoz_weight'

    fig = plt.figure(figsize=(7, 6))
    fig.subplots_adjust(left=0.18, right=0.95,
                        bottom=0.15, top=0.95,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111, title=qa_title)

    ax1.plot(zbin_center, z_weights, linewidth=3)

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax1.set_xlabel(r'$\mathrm{Photometric\ redshift}$', fontsize=16)
    ax1.set_ylabel(r'$\mathrm{Weights}$', fontsize=16)

    fig.savefig(qa_prefix + '.png', dpi=90)


def qa_lens_rand_zhist(lens_z, rand_z, rand_zweight,
                       qa_title='Redshift distributions: lens-random',
                       qa_prefix=None):
    """Plot the reweighted redshift distribution."""
    if qa_prefix is None:
        qa_prefix = 'qa_rand_photoz_hist'
    else:
        qa_prefix += '_qa_rand_photoz_hist'

    fig = plt.figure(figsize=(6, 5))
    fig.subplots_adjust(left=0.09, right=0.98,
                        bottom=0.15, top=0.95,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111, title=qa_title)

    zrange = (np.nanmin(rand_z), np.nanmax(rand_z))
    ax1.hist(lens_z, bins=15, density=1, range=zrange,
             label='lens', facecolor='k', alpha=0.4)
    ax1.hist(rand_z, bins=15, alpha=0.2, range=zrange,
             density=1, weights=rand_zweight,
             label='random', facecolor='b')
    ax1.hist(rand_z, bins=15, alpha=0.8, range=zrange,
             density=1, weights=rand_zweight, histtype='step',
             linewidth=3, label='random (reweighted)',
             edgecolor='r')

    ax1.grid(linestyle='--', alpha=0.2, linewidth=2)
    ax1.legend(fontsize=12, loc=2)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax1.set_xlabel(r'$\mathrm{Photometric\ redshift}$', fontsize=16)

    # plt.show()
    fig.savefig(qa_prefix + '.png', dpi=90)


def plot_corr_matrix(log_rmpc, corr_matrix, qa_prefix=None):
    """Plot the covariance matrix."""
    if qa_prefix is None:
        qa_prefix = 'qa_corr_matrix'
    else:
        qa_prefix += '_qa_corr_matrix'

    fig = plt.figure(figsize=(7.5, 7))
    fig.subplots_adjust(left=0.16, right=0.995,
                        bottom=0.12, top=0.995,
                        wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)

    cax1 = ax1.pcolor(log_rmpc, log_rmpc, corr_matrix, cmap='Oranges')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    ax1.set_xlabel(r'$\log\ (\mathrm{R}/\mathrm{Mpc})$', fontsize=30)
    ax1.set_ylabel(r'$\log\ (\mathrm{R}/\mathrm{Mpc})$', fontsize=30)

    cbaxes1 = fig.add_axes([0.90, 0.20, 0.03, 0.33])
    _ = fig.colorbar(cax1, cax=cbaxes1, ticks=[-0.1, 0.1, 0.5, 0.8, 1.0])

    fig.savefig(qa_prefix + '.png', dpi=100)


def show_delta_sigma_profiles(list_prof, list_label=None,
                              ax=None, m_list=None, c_list=None,
                              s_list=None, a_list=None,
                              label_size=25, legend_size=10,
                              ymin_force=None, ymax_force=None):
    """Show a list of delta sigma profiles."""
    if m_list is None:
        m_list = ['o', 's', 'D', 'P', 'h', '8', 'p', '<', '>', 'v']
    if c_list is None:
        c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    if s_list is None:
        s_list = [50, 60, 50, 50, 50, 50, 50, 50, 50, 50]
    if a_list is None:
        a_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        fig.subplots_adjust(left=0.18, right=0.99,
                            bottom=0.13, top=0.95,
                            wspace=0.00, hspace=0.00)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    if list_label is None:
        list_label = ['__no_label__'] * len(list_prof)
    else:
        assert len(list_prof) == len(list_label)

    ymax = 100.0
    ymin = 2.0
    for ii, sig in enumerate(list_prof):
        radial_bins = sig['r_mpc']
        delta_sigma = sig['dsigma_lr']
        errors = sig['dsigma_err_jk']

        if np.nanmax(delta_sigma) > ymax:
            ymax = np.nanmax(delta_sigma)
        if np.nanmin(delta_sigma) < ymin:
            ymin = np.nanmin(delta_sigma)

        x_offset = 1.0 + 0.01 * ii
        ax1.scatter(radial_bins * x_offset,
                    delta_sigma,
                    s=s_list[ii],
                    alpha=a_list[ii],
                    marker=m_list[ii],
                    c=c_list[ii],
                    label=list_label[ii])

        if errors is not None:
            ax1.errorbar(radial_bins * x_offset,
                         delta_sigma,
                         yerr=errors,
                         color=c_list[ii],
                         ecolor=c_list[ii],
                         fmt=m_list[ii],
                         alpha=0.3,
                         capsize=4,
                         capthick=1,
                         elinewidth=1,
                         label='__no_label__')

    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax1.legend(loc='best', fontsize=legend_size)

    ax1.set_xlim(radial_bins.min() * 0.6, radial_bins.max() * 1.5)

    if ymin_force is not None:
        ymin = ymin_force
    if ymax_force is not None:
        ymax = ymax_force

    ax1.set_ylim(ymin * 0.5, ymax * 2.0)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax1.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=label_size)
    ax1.set_ylabel(r'$\Delta\Sigma\ (M_{\odot}/\mathrm{pc}^2)$',
                   fontsize=label_size)

    if ax is None:
        return fig

    return ax


def show_r_delta_sigma(list_prof, list_label=None, ax=None,
                       m_list=None, c_list=None,
                       s_list=None, a_list=None,
                       label_size=25, legend_size=10):
    """Show a list of delta sigma profiles."""
    if m_list is None:
        m_list = ['o', 's', 'D', 'P', 'h', '8', 'p', '<', '>', 'v']
    if c_list is None:
        c_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    if s_list is None:
        s_list = [50, 60, 50, 50, 50, 50, 50, 50, 50, 50]
    if a_list is None:
        a_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3]

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        fig.subplots_adjust(left=0.18, right=0.99,
                            bottom=0.13, top=0.95,
                            wspace=0.00, hspace=0.00)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax

    ax1.set_xscale("log", nonposx='clip')
    # ax1.set_yscale("log", nonposy='clip')

    if list_label is None:
        list_label = ['__no_label__'] * len(list_prof)
    else:
        assert len(list_prof) == len(list_label)

    ymax = 30.0
    ymin = 5.0
    for ii, sig in enumerate(list_prof):
        radial_bins = sig['r_mpc']
        delta_sigma = sig['dsigma_lr']
        errors = sig['dsigma_err_jk']

        if np.nanmax(radial_bins * delta_sigma) > ymax:
            ymax = np.nanmax(delta_sigma * radial_bins)
        if np.nanmin(radial_bins * delta_sigma) < ymin:
            ymin = np.nanmin(delta_sigma * radial_bins)

        x_offset = 1.0 + 0.01 * ii
        ax1.scatter(radial_bins * x_offset,
                    radial_bins * delta_sigma,
                    s=s_list[ii],
                    alpha=a_list[ii],
                    marker=m_list[ii],
                    c=c_list[ii],
                    label=list_label[ii])

        if errors is not None:
            ax1.errorbar(radial_bins * x_offset,
                         radial_bins * delta_sigma,
                         yerr=errors,
                         color=c_list[ii],
                         ecolor=c_list[ii],
                         fmt=m_list[ii],
                         alpha=0.3,
                         capsize=4,
                         capthick=1,
                         elinewidth=1,
                         label='__no_label__')

    ax1.grid(linestyle='--', alpha=0.4, linewidth=2)
    ax1.legend(loc='best', fontsize=legend_size)

    ax1.set_xlim(radial_bins.min() * 0.6, radial_bins.max() * 1.5)
    ax1.set_ylim(ymin * 0.8, ymax * 1.4)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    ax1.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=label_size)
    ax1.set_ylabel(r'$R \times \Delta\Sigma\ (M_{\odot}/\mathrm{pc}^2)$',
                   fontsize=label_size)

    if ax is None:
        return fig

    return ax


def show_single_profile(rmpc, dsigma, dsigma_err=None, ax=None, label='',
                        alpha=0.9, s=20, c='b', marker='o', label_size=25):
    """Show a list of delta sigma profiles."""
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        fig.subplots_adjust(left=0.18, right=0.99,
                            bottom=0.13, top=0.95,
                            wspace=0.00, hspace=0.00)
        ax1 = fig.add_subplot(111)
        ax1.grid(linestyle='--', alpha=0.4, linewidth=2)

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)

        ax1.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=label_size)
        ax1.set_ylabel(r'$\Delta\Sigma\ (M_{\odot}/\mathrm{pc}^2)$',
                       fontsize=label_size)
    else:
        ax1 = ax

    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')

    ymin, ymax = np.nanmin(dsigma), np.nanmax(dsigma)

    if dsigma_err is not None:
        ax1.errorbar(rmpc, dsigma, yerr=dsigma_err, color=c, ecolor=c, fmt=marker,
                     alpha=alpha, capsize=4, capthick=1, elinewidth=1.5, 
                     zorder=1, label='__no_label__')

    ax1.scatter(rmpc, dsigma, s=s, facecolor=c, edgecolor='k', alpha=alpha,
                marker=marker, label=label, zorder=20)

    ax1.set_ylim(ymin * 0.8, ymax * 2.0)

    if ax is None:
        return fig

    return ax
