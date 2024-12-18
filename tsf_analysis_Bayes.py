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

sns.set_palette("hls",4)

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
freqs = np.arange(20, 100, 0.03125)
Tvals = [168* 24 * 60 * 60,      365.25* 24 * 60 * 60,       365.25*2* 24 * 60 * 60, 
        365.25*5* 24 * 60 * 60,        365.25*10* 24* 60 * 60, 365.25*20* 24 * 60 * 60, 
        365.25*50* 24 * 60 * 60,       365.25*100* 24 * 60 * 60] 

Tvals = [7* 24 * 60 * 60, 14* 24 * 60 * 60, 30* 24 * 60 * 60,      
         168* 24 * 60 * 60,      365.25* 24 * 60 * 60,       365.25*2* 24 * 60 * 60, 
        365.25*5* 24 * 60 * 60,        365.25*10* 24* 60 * 60, 365.25*20* 24 * 60 * 60, 
        365.25*50* 24 * 60 * 60,      365.25*100* 24 * 60 * 60, 365.25*500* 24 * 60 * 60, 365.25*1000* 24 * 60 * 60]   

N_samples = 500_000
N_offset = 10_000

# generate data given a keyword ['BPL', 'squiggly', 'Sachdev'] - currently using a txt file for the Sachdev curve
sig_type = 'FOPT'
noise = 'CE'

Bayes1 = []
T1 = []
Bayes2 = []
T2 = []
Bayes3 = []
T3 = []

for kk, T in enumerate(Tvals): 
    print(f'TIME: {T}')
    try:
        freqs, data, signal, fit_omega, fit_results_omega = pickle.load(open(f'{sig_type}_signal_data_{T/(24 * 60 * 60)}_{noise}.pkl', 'rb'))

        plt.clf()
        
        print('Bayes Factors \n --------------')
        configs, num_knots = return_knot_info(fit_results_omega, offset=N_offset)
        
        print(f'Configs: \n {configs}')
        print(f'Num Knots: \n {num_knots}')
        
        try:
            print(f'Bayes Factor Detection: {len([i for i in num_knots if i > 0]) / len([i for i in num_knots if i == 0])}')
            Bayes1.append(len([i for i in num_knots if i > 0]) / len([i for i in num_knots if i == 0]))
            T1.append(T/(365.25* 24 * 60 * 60))
        except: 
            print('DIVIDE BY ZERO ERROR')
            Bayes1.append(np.inf)
            T1.append(T/(365.25* 24 * 60 * 60))
        try:
            print(f'Bayes Factor BPL v. PL: {len([i for i in num_knots if i >= 3]) / len([i for i in num_knots if i == 1 or i == 2])}')
            Bayes2.append(len([i for i in num_knots if i >= 3]) / len([i for i in num_knots if i == 1 or i == 2]))
            T2.append(T/(365.25* 24 * 60 * 60))
        except: 
            print('DIVIDE BY ZERO ERROR')
            Bayes2.append(np.inf)
            T2.append(T/(365.25* 24 * 60 * 60))
        try:
            print(f'Bayes Factor Slope: {len([i for i in num_knots if i == 2]) / len([i for i in num_knots if i == 1])}')
            Bayes3.append(len([i for i in num_knots if i == 2]) / len([i for i in num_knots if i == 1]))
            T3.append(T/(365.25* 24 * 60 * 60))
        except: 
            print('DIVIDE BY ZERO ERROR')
            Bayes3.append(np.inf)
            T3.append(T/(365.25* 24 * 60 * 60))
    except: print('I forgor')
    '''
    # plot histogram of number of knots turned on
    bins, ed, ___ = plt.hist(num_knots, bins=np.linspace(-0.5, 30.5, num=32), edgecolor='w')

    print(f'Bins: {bins}')
    print(f'Edges: {ed}')

    plt.axvline(np.average(num_knots), label = 'avg = '+str(np.average(num_knots)), linestyle='--', c='black')
    plt.xticks(np.arange(0, 30))
    plt.yscale("log")
    plt.xlim(-1, 10)
    plt.xlabel('num knots')
    plt.legend()
    plt.show()
    plt.savefig(f'tsf_results/tsf_knot_hist_{T/(24 * 60 * 60)}_HL.pdf')
    plt.clf()
    print('----------------------------')
    '''

plt.loglog(T1, Bayes1, label = 'Bayes - Detection', marker = '.', markersize='10')
plt.loglog(T2, Bayes2, label = 'Bayes - BPL v. PL', marker = '.', markersize='10')
plt.loglog(T3, Bayes3, label = 'Bayes - Slope', marker = '.', markersize='10')
plt.legend()
plt.xlabel('Observing Time [yrs]')
plt.ylabel('Bayes Factor')
plt.tight_layout()
plt.savefig(f'BayesFactors_{sig_type}_{noise}.pdf')

print(f'Bayes 1: {Bayes1}')
print(f'Bayes 2: {Bayes2}')
print(f'Bayes 3: {Bayes3}')

