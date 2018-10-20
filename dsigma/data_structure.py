"""Useful fields for shape catalog."""

import numpy as np

__all__ = ['get_source_fields', 'get_lens_fields', 'get_ds_out_field',
           'get_results_arr', 'get_ds_output_array']


def get_source_fields(version):
    """Make accessing fields in our source file a bit easier.

    No-one wants to type `ishape_hsm_regauss_derived_shear_bias_m` just to get `bias_m`.

    Parameters
    ----------
    version : str
        String that indicates the source catalog used for analysis.

    Return
    ------
    source_fields : dict
        Useful fields from the source catalog
    """
    if version == 's16a':
        return {
            'ra': 'ira',
            'dec': 'idec',
            'field': 'field',
            'z': '',      # Redshift; see config file
            'z_err': '',  # Uncertaintiy of the redshift; see config file
            'z_low': '',  # Lower bound of the distribution function of redshift; see config file
            'e1': 'ishape_hsm_regauss_e1',
            'e2': 'ishape_hsm_regauss_e2',
            'weight': 'ishape_hsm_regauss_derived_shape_weight',
            'bias_m': 'ishape_hsm_regauss_derived_shear_bias_m',
            'e_rms': 'ishape_hsm_regauss_derived_rms_e',
            'r2': 'ishape_hsm_regauss_resolution',
            }
    else:
        # This is a placeholder
        # Right now we only have S16A source catalog
        raise Exception("# Wrong type of source catalog: [s16a]")

def get_lens_fields(cfg):
    """Return a dictionary of useful fields in the lens catalog.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for pre-compute process.

    Return
    ------
    lens_fields: dict
        Shortcuts for useful fields in the lens catalog.
    """
    if 'lens' in cfg:
        if 'ra' not in cfg['lens']:
            cfg['lens']['ra'] = 'ra'

        if 'dec' not in cfg['lens']:
            cfg['lens']['dec'] = 'dec'

        if 'z' not in cfg['lens']:
            cfg['lens']['z'] = 'z'

        if 'field' not in cfg['lens']:
            cfg['lens']['field'] = 'field'

        if 'weight' not in cfg['lens']:
            cfg['lens']['weight'] = 'weight'
        print("# Using column %s as lens weight" % cfg['lens']['weight'])

        return {
            'ra': cfg['lens']['ra'],
            'dec': cfg['lens']['dec'],
            'z': cfg['lens']['z'],
            'field': cfg['lens']['field'],
            'weight': cfg['lens']['weight'],
        }

    return {'ra': 'ra', 'dec': 'dec', 'z': 'z', 'field': 'field', 'weight': 'weight'}


def get_results_arr(config):
    """Return an array to store the per-pair results."""
    return np.zeros(1, dtype=[
        ("field", "int"),
        ("ra", "float64"),
        ("dec", "float64"),
        ("z", "float64"),
        ("weight", "float64"),
        ("zp_bias_num", "float64"),
        ("zp_bias_den", "float64"),
        ("radial_bins", [
            ("sum_num_t", "float64"),  # numerator of tangential term
            ("sum_num_x", "float64"),  # numerator of cross term
            ("sum_num_t_sq", "float64"),  # for shape error
            ("sum_num_x_sq", "float64"),  # for shape error
            ("sum_den", "float64"),    # denominator of delta_sigma
            ("sum_num_k", "float64"),  # numerator of the k term
            ("sum_num_r", "float64"),  # numerator of the R-1 term
            ("sum_num_m_sel", "float64"),  # numerator for the selection bias term
            ("sum_den_m_sel", "float64"),  # denominator for the selection bias term
            ("ds_t_sq", "float64"),  # square terms for tangential shape noise
            ("ds_x_sq", "float64"),  # square terms for cross shape noise
            ("n_pairs", "int64"),     # number of pairs in each radius bin
            ("sum_dist", "float64"), # sum of the lens-sources distance in each radius bin
            ("e_weight", "float64"),
            ], (config['binning']['nbins'])),
        ])[0]


def get_ds_output_array(radial_bins):
    """Return a zero array for output.

    Parameters
    ----------
    radial_bins : numpy array
        1-D array for radial bins.

    Return
    ------
        Empty structured array for computeDS output.

    Notes
    -----
        - r_mpc : radial bin center in Mpc
        - dsigma_lr : fiducial delta_sigma signal, with random subtraction.
        - dsigma_err_1: simply variance of the weighted mean.
        - dsigma_err_2: delta_sigma error based on Melanie's gglens.
        - dsigma_err_jk: jackknife delta_sigma error.
        - lens_dsigma_t: tangential delta_sigma for lens without any correction.
        - lens_dsigma_x: cross delta_sigma for lens without any correction.
        - lens_r: responsivity correction factor for lens.
        - lens_k: multiplicative bias for lens.
        - lens_m: resolution selection bias for lens.
        - lens_calib: calibration factors for lens.
        - rand_dsigma_t: tangential delta_sigma for random without any correction.
        - rand_dsigma_x: cross delta_sigma for random without any correction.
        - rand_r: responsivity correction factor for random.
        - rand_k: multiplicative bias for random.
        - rand_m: resolution selection bias for random.
        - rand_calib: calibration factors for random.
        - boost_factor: boost correction factor for lens.
        - lens_npairs: number of lens-source pairs in each radial bin.
        - rand_npairs: number of random-source pairs in each radial bin.
        - rand_npairs_eff: number of random-source pairs after re-weight.
    """
    dsig = np.zeros(len(radial_bins),
                    dtype=[("r_mpc", "float64"),
                           ("r_mean_mpc", "float64"),
                           ("dsigma_lr", "float64"),
                           ("dsigma_err_1", "float64"),
                           ("dsigma_err_2", "float64"),
                           ("dsigma_err_jk", "float64"),
                           ("lens_dsigma_t", "float64"),
                           ("lens_dsigma_x", "float64"),
                           ("lens_r", "float64"),
                           ("lens_k", "float64"),
                           ("lens_m", "float64"),
                           ("lens_calib", "float64"),
                           ("r_mean_mpc_rand", "float64"),
                           ("rand_dsigma_t", "float64"),
                           ("rand_dsigma_x", "float64"),
                           ("rand_r", "float64"),
                           ("rand_k", "float64"),
                           ("rand_m", "float64"),
                           ("rand_calib", "float64"),
                           ("boost_factor", "float64"),
                           ("zp_bias_lens", "float64"),
                           ("zp_bias_rand", "float64"),
                           ("lens_npairs", "int64"),
                           ("lens_npairs_eff", "int64"),
                           ("rand_npairs", "int64"),
                           ("rand_npairs_eff", "int64")])

    dsig['r_mpc'] = radial_bins

    return dsig


def get_ds_out_field(n_bins):
    """Return an empty structure array for the DeltaSigma result in one field.

    Parameters
    ----------
    n_bins : int
        Number of radial bins.

    Return
    ------
        Empty structured array for computeDS output.

    Notes
    -----

    For lens and random, the tangential DeltaSigma sigma is:

        ds_t_lens = ds_t_num_lens / ds_den_lens
        ds_t_rand = ds_t_num_rand / ds_den_rand

    The cross term signal is:

        ds_x_lens = ds_x_num_lens / ds_den_lens
        ds_x_rand = ds_x_num_rand / ds_den_rand

    The R-term in te calibration factor is:

        r_lens = r_num_lens / ds_den_lens
        r_rand = r_num_rand / ds_den_rand

    The k-term in the calibration factor is:

        k_lens = k_num_lens / ds_den_lens
        k_rand = k_num_rand / ds_den_rand

    The optional m-term in the calibration factor is:

        m_lens = m_num_lens / m_den_lens
        m_rand = m_num_rand / m_den_rand

    The optional boost correction factor is:

        boost = (ds_den_lens * w_rand) / (ds_den_rand * w_lens)

    The final calibration term is:

        calib_lens = 1. / (2 * r_lens * (1 + k_lens) * (1 + m_lens))
        calib_rand = 1. / (2 * r_rand * (1 + k_rand) * (1 + m_rand))

    The final DeltaSigma signal afte correction:

        dsigma = (ds_t_lens * calib_lens * boost - ds_t_rand * calib_rand) * zp_bias

    """
    dsig = np.zeros(n_bins,
                    dtype=[("ds_t_num_lens", "float64"),
                           ("ds_x_num_lens", "float64"),
                           ("ds_t_num_rand", "float64"),
                           ("ds_x_num_rand", "float64"),
                           ("ds_den_lens", "float64"),
                           ("ds_den_rand", "float64"),
                           ("r_num_lens", "float64"),
                           ("r_num_rand", "float64"),
                           ("m_num_lens", "float64"),
                           ("m_num_rand", "float64"),
                           ("m_den_lens", "float64"),
                           ("m_den_rand", "float64"),
                           ("k_num_lens", "float64"),
                           ("k_num_rand", "float64"),
                           ("w_lens", "float64"),
                           ("w_rand", "float64"),
                           ("lens_npairs", "int64"),
                           ("rand_npairs", "int64"),
                           ])

    # Fill the denominators with 1.0 so that even empty array will give finite results
    dsig['ds_den_lens'] = np.ones(n_bins)
    dsig['ds_den_rand'] = np.ones(n_bins)
    dsig['m_den_lens'] = np.ones(n_bins)
    dsig['m_den_rand'] = np.ones(n_bins)
    dsig['w_lens'] = np.ones(n_bins)
    dsig['w_rand'] = np.ones(n_bins)
    dsig['r_num_lens'] = np.ones(n_bins)
    dsig['r_num_rand'] = np.ones(n_bins)

    return dsig
