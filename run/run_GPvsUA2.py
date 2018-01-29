#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp#, ExpSquaredCenteredKernel, ExpSquaredKernel
from iminuit import Minuit
import scipy.special as ssp
import inspect
from lib-plotmeghan import *

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    return parser.parse_args()

def run_vs_UA2Fit():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points/fit
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')
    #xSig, ySig, xerrSig, yerrSig = getDataPoints(args.signal_file, 'dijetgamma_g150_2j25', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    x, y, xerr, yerr = dataCut(300, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xMinFit=300
    xFit, yFit, xerrFit, yerrFit= dataCut(xMinFit, 1500, 0, xRaw, yRaw,xerrRaw,yerrRaw)

    # make an evently spaced x
    t = np.linspace(np.min(x), np.max(x), 500)

    #calculating the log-likihood and minimizing for the gp
    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(100, lnProb)

    fit_pars = [best_fit[x] for x in FIT3_PARS]

    #making the GP
    kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    #making the kernel
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    # calculating the fit function value

    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]
    print ("best param meghan GP minmization:", best)
    meanFromGPFit=Mean(best)

    fit_meanM =meanFromGPFit.get_value(x,xerr) #so this is currently shit. can't get the xErr thing working
    print("fit_meanM:",fit_meanM)

    #fit_mean_smooth = gp_new.mean.get_value(t)

    #----3 param fit function in a different way
    lnProbUA2 = logLike_UA2(xFit,yFit,xerrFit)
    minimumLLH, best_fit_params = fit_UA2(100, lnProbUA2)
    fit_mean = model_UA2(xFit, best_fit_params, xerrFit)

    ##----4 param fit function
    #lnProb = logLike_4ff(xFit,yFit,xerrFit)
    #minimumLLH, best_fit_params = fit_4ff(100, lnProb)
    #fit_mean = model_4param(xFit, best_fit_params, xerrFit)


    #calculating significance
    signif = significance(mu_x, y, cov_x, yerr)

    initialCutPos=np.argmax(x>xMinFit)

    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    sigFit =significance(fit_mean, y[initialCutPos:], cov_x[initialCutPos:], yerr[initialCutPos:])
    #sigit = (mu_x)
    #std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    title="compareGPvs3UA2New"
    with Canvas(f'%s{ext}'%title) as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        can.ax.plot(xFit, fit_mean, '-b')
        #this only works with the xErr part of Mean commented out
        #can.ax.plot(x, fit_meanM, '.g')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        #can.ax.fill_between(t, mu - std, mu + std,
                            #facecolor=(0, 1, 0, 0.5),
                            #zorder=5, label='err = 1')
        #can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.stem(xFit, sigFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)
