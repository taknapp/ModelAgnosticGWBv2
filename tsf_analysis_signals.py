# PLOT RECOVERED OMEGA_GW
import sys
import transdimensional_spline_fitting as tsf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm
from scipy.optimize import minimize
from functools import partial
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d

mpl.rcParams.update(mpl.rcParamsDefault)

import pygwb
import bilby
import astropy.cosmology
from copy import deepcopy
from pygwb.baseline import Baseline
import seaborn as sns

from popstock_tsf_helper import *

# Signficantly speeds things up
import lal
lal.swig_redirect_standard_output_error(False)

R0 = 31.4
H0 = astropy.cosmology.Planck18.H0.to(astropy.units.s**-1).value

class SmoothCurveDataObj(object):
    """
    A data class that can be used with our spline model
    """
    def __init__(self, data_xvals, data_yvals, data_errors):
        self.data_xvals = data_xvals
        self.data_yvals = data_yvals
        self.data_errors = data_errors

class FitRedshift(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """
    def ln_likelihood(self, config, heights):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        redshift_model = 10**self.evaluate_interp_model(np.log10(bbh_pickle.ref_zs), heights, config, log_xvals=True)
        
        model = bbh_pickle.eval(R0, redshift_model, self.data.data_xvals)
        
        return np.sum(norm.logpdf(model - self.data.data_yvals, scale=self.data.data_errors))

class FitOmega(tsf.BaseSplineModel):
    """
    Example of subclassing `BaseSplineModel` to create a likelihood
    that can then be used for sampling.

    Assumes use with `ArbitraryCurveDataObj`

    You also need to create a simple data class to go along with this. This
    allows the sampler to be used with arbitrary forms of data...
    """

    
    def ln_likelihood(self, config, heights, knots):
        """
        Simple Gaussian log likelihood where the data are just simply
        points in 2D space that we're trying to fit.

        This could be something more complicated, though, of course. For example,
        You might create your model from the splines (`model`, below) and then use that
        in some other calculation to put it into the space for the data you have.

        :param data_obj: `ArbtraryCurveDataObj` -- an instance of the data object class associated with this likelihood.
        :return: log likelihood
        """
        # be careful of `evaluate_interp_model` function! it does require you to give a list of xvalues,
        # which don't exist in the base class!
        omega_model = 10**self.evaluate_interp_model(np.log10(self.data.data_xvals), heights, config, np.log10(knots))
        # print(omega_model)
        # print(self.data.data_yvals)
        # print(self.data.data_errors)

        return np.sum(norm.logpdf(omega_model - self.data.data_yvals, scale=self.data.data_errors))

# VARIABLES   
freqs = np.arange(12, 100, 0.03125)
Tvals = [7* 24 * 60 * 60, 14* 24 * 60 * 60, 30* 24 * 60 * 60,      
         168* 24 * 60 * 60,      365.25* 24 * 60 * 60,       365.25*2* 24 * 60 * 60, 
        365.25*5* 24 * 60 * 60,        365.25*10* 24* 60 * 60, 365.25*20* 24 * 60 * 60, 
        365.25*50* 24 * 60 * 60,       365.25*100* 24 * 60 * 60]   

Tvals = [168* 24 * 60 * 60,      365.25* 24 * 60 * 60,       365.25*2* 24 * 60 * 60, 
        365.25*5* 24 * 60 * 60,        365.25*10* 24* 60 * 60, 365.25*20* 24 * 60 * 60, 
        365.25*50* 24 * 60 * 60,       365.25*100* 24 * 60 * 60]

N_samples = 100_000
N_offset = 10_000

sig_type = 'classic'
noise_type = 'CE'

for kk, T in enumerate(Tvals): 
    print(f'TIME: {T}')

    freqs, data, signal, fit_omega, fit_results_omega = pickle.load(open(f'{sig_type}_signal_data_{T/(24 * 60 * 60)}_{noise_type}.pkl', 'rb'))
    plot_posterior_fits(fit_omega, fit_results_omega, freqs, N_samples, offset=N_offset, num_posteriors=200, label_str = '95\%')

    plt.plot(freqs, data, alpha=0.3, label='Data', zorder=1000)
    plt.plot(freqs, signal, c='r', ls='--', label='True signal', zorder=1001)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-13, 1e-5)
    plt.xlabel('freqs [Hz]')
    plt.ylabel('$\Omega_{GW}$')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'tsf_posterior_fits_{T/(24 * 60 * 60)}_{noise_type}_{sig_type}.pdf')
    plt.clf()
