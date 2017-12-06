#!/usr/bin/env python3
#using the bkgnd data of MC reweighted of dijetgamma_g85_2j65
from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp#, ExpSquaredCenteredKernel#, ExpSquaredKernel
from iminuit import Minuit
import scipy.special as ssp
import inspect
from Libplotmeghan import getDataPoints,dataCut, y_bestFit3Params, y_bestFitGP, res_significance, significance

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    return parser.parse_args()


def run_bkgnd():
    xMin=300
    xMax=1500
    xMinFit=300
    xMaxFit=1500

    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, '', 'MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1')

    #Data processing, cutting out the not desired range
    xBkg, yBkg, xerrBkg, yerrBkg = dataCut(xMin, xMax, 0, xRaw, yRaw, xerrRaw, yerrRaw) # for GP bkgnd kernel and signal kernel 
    xBkgFit, yBkgFit, xerrBkgFit, yerrBkgFit= dataCut(xMinFit, xMaxFit, 0, xRaw, yRaw,xerrRaw,yerrRaw) # for fit function 

    # make an evently spaced x
    t = np.linspace(np.min(xBkg), np.max(xBkg), 500)

    #Drawing the raw data of just the bkgnd and Sig + bkgnd
    with Canvas(f'RawSig') as can: # just to check and see everything is okay
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g')
        can.ax.set_yscale('log')

    #GP bkgnd: finidng the mean and covaraicne 
    mu_xBkg, cov_xBkg=y_bestFitGP(xBkg,yBkg,xerrBkg, yerrBkg, 3, kernelType="bkg")

    # GP Signal: finind the mean of covariance 
    mu_xSig, cov_xSig= y_bestFitGP(xBkg, yBkg, xerrBkg, yerrBkg, 30, kernelType="sig")

#finding the fit y values 
    fit_mean=y_bestFit3Params(xBkgFit, yBkgFit, xerrBkgFit, 3)

#finding signifiance 
    GPSignificance, chi2=res_significance(yBkg, mu_xBkg)
    fitSignificance, chi2fit=res_significance(yBkgFit, fit_mean)
    GPSigSignificance, chi2SignalFit=res_significance(yBkg, mu_xSig)

#drawing the result
    ext = args.output_file_extension
    title="test"
    with Canvas(f'%s{ext}'%title, "Fit Function", "GP bkgnd kernel", "GP signal+bkgnd kernel", 3) as can:
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g') # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(xBkg, mu_xBkg, '-g', label="unicorn") #drawing 
        can.ax.plot(xBkgFit, fit_mean, '-r')
        can.ax.plot(xBkg, mu_xSig, '-b', label="GP bkg kernel pony") 
        can.ax.legend(framealpha=0)
        can.ratio.stem(xBkgFit, fitSignificance, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xBkgFit, testsig, markerfmt='.', basefmt=' ')
        can.ratio.set_ylabel("significance")
        can.ratio2.stem(xBkg, GPSignificance, markerfmt='.', basefmt=' ')
        can.ratio2.set_ylabel("significance")
        can.ratio3.set_ylabel("significance")
        can.ratio3.stem(xBkg, GPSigSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.ratio3.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

#calculting with the signal kernel -
'''
    lnProbSig = logLike_gp_fitgpsig(xSig, ySig, xerrSig)
    min_likelihoodSig, best_fitSig = fit_gp_fitgpsig_minuit(lnProbSig)

    fit_pars = [best_fit[xSig] for xSig in FIT3_PARS]

    Args=kargs+best_fitSig
    kernel_Sig = get_kernelXtreme(**Arg)
    print(kernel_Sig.get_parameter_names())
    #making the kernel
    gp_sig = george.GP(kernel_Sig, mean=Mean(fit_pars), fit_mean = True)
    gp_sig.compute(xSig, yerrSig)
    muSig, covSig = gp_sig.predict(ySig, t)
    mu_xSig, cov_xSig = gp_Sig.predict(ySig, xSig)
    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]
#calculating significance
    signBkg = significance(mu_xBkg, yBkg, cov_xBkg, yerrBkg)


    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    signSig =significance(mu_xSig, ySig, cov_xSig, yerrSig)
'''
if __name__ == '__main__':
    run_bkgnd()
