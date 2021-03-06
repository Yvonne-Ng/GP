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
from Libplotmeghan import getDataPoints,dataCut, y_bestFit3Params, y_bestFitGP, res_significance, significance, psuedoTest, logLike_3ffOff
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('fitFromOfficial')
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
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')
    #data points from official fit function
    xFitRaw, yFitRaw, xerrFitRaw, yerrFitRaw = getDataPoints(args.fitFromOfficial, "", "basicBkgFrom4ParamFit")
    print("xFit: ", xFitRaw)

    #Data processing, cutting out the not desired range
    xBkg, yBkg, xerrBkg, yerrBkg = dataCut(xMin, xMax, 0, xRaw, yRaw, xerrRaw, yerrRaw) # for GP bkgnd kernel and signal kernel 
    xBkgFit, yBkgFit, xerrBkgFit, yerrBkgFit= dataCut(xMinFit, xMaxFit, 0, xRaw, yRaw,xerrRaw,yerrRaw) # for fit function 
    xFit, yFit, xerrFit, yerrFit = dataCut(xMinFit, xMaxFit, 0, xFitRaw, yFitRaw, xerrFitRaw, yerrRaw)

    # calculate the min log likelihood of the official fit function
    logLikeOff=logLike_3ffOff(xBkg, yBkg, xFit, yFit, xerrBkg)
    # make an evently spaced x
    t = np.linspace(np.min(xBkg), np.max(xBkg), 500)

    #Drawing the raw data of just the bkgnd and Sig + bkgnd
    with Canvas(f'RawBkgnd') as can: # just to check and see everything is okay
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g')
        can.ax.set_yscale('log')

    #GP bkgnd: finidng the mean and covaraicne 
    mu_xBkg, cov_xBkg, bestFitBkg=y_bestFitGP(xBkg,yBkg,xerrBkg, yerrBkg, 30, kernelType="bkg")


    # GP Signal: finind the mean of covariance 
    mu_xSig, cov_xSig, bestFitSig= y_bestFitGP(xBkg, yBkg, xerrBkg, yerrBkg, 30, kernelType="sig")

#finding the fit y values 
    fit_mean=y_bestFit3Params(xBkgFit, yBkgFit, xerrBkgFit, 1)

    #Signal only fit
    ySigFit= mu_xSig-mu_xBkg
    print("ySigFit: ",ySigFit)

#finding signifiance 
    GPSignificance, chi2=res_significance(yBkg, mu_xBkg)
    fitSignificance, chi2fit=res_significance(yBkgFit, fit_mean)
    GPSigSignificance, chi2SignalFit=res_significance(yBkg, mu_xSig)
    FitSigFromOff, chi2FitOff = res_significance(yBkgFit,yFit)
#drawing the result
    ext = args.output_file_extension
    title="test"
    with Canvas(f'%s{ext}'%title, "Fit Function official ", "GP bkgnd kernel", "GP signal+bkgnd kernel", 3) as can:
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g') # drawing the points
        can.ax.set_yscale('log')
        #can.ax.plot(xBkgFit, fit_mean, '-r', label="fit function")
        can.ax.plot(xFit, yFit, '-r', label="fit function official")
        can.ax.plot(xBkg, mu_xBkg, '-g', label="GP bkg kernel") #drawing 
        can.ax.plot(xBkg, mu_xSig, '-b', label="GP signal kernel") 
        can.ax.legend(framealpha=0)
        can.ratio.stem(xFit, FitSigFromOff, markerfmt='.', basefmt=' ')
        can.ratio.set_ylabel("significance")
        can.ratio2.stem(xBkg, GPSignificance, markerfmt='.', basefmt=' ')
        can.ratio2.set_ylabel("significance")
        can.ratio3.set_ylabel("significance")
        can.ratio3.stem(xBkg, GPSigSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.ratio3.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

    fitChi2List, GPChi2List=psuedoTest(100, yBkgFit,yerrBkgFit, fit_mean, yBkg,yerrBkg,  mu_xBkg)
    print("fitChi2List", fitChi2List)
    print("GPChi2List", GPChi2List)
    #n, bins, patches = plt.hist(fitChi2List, 50, normed=1, facecolor='green', alpha=0.75)
    #plt.plot(n, drawstyle="steps")
    #plt.savefig("chi2.pdf")

    value, edges = np.histogram(fitChi2List)
    print("value %r, edges %r" %(value, edges))
    binCenters=(edges[1:]+edges[:-1])/2
    valueGP, edgesGP = np.histogram(GPChi2List)
    print("value %r, edges %r" %(valueGP, edgesGP))
    binCentersGP=(edgesGP[1:]+edgesGP[:-1])/2
    #plt.step(binCenters,value)
    #plt.savefig("chi2.pdf")

    n=binCenters.size-1
    nGP=binCentersGP.size-1
    
    title="chi2"
    with Canvas(f'%s{ext}'%title, "Fit Function", "GP bkgnd kernel", "", 2) as can:
        can.ax.step(binCenters/n, value, label="Fit Function",where="mid")
        can.ax.step(binCentersGP/nGP, valueGP, label="GP bkgnd", where="mid")
        can.ax.legend(framealpha=0)
        can.save(title)
    print("min LL of off: ",logLikeOff)

#drawing the signal only 
    ext = args.output_file_extension
    title="signalonlybkg"
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        #can.ax.errorbar(xBkg, ySig, yerr=yerrBkg, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(-1000,1000.0)
        #can.ax.set_yscale('log')
        can.ax.plot(xBkg, ySigFit, '-r', label="GP Sig + bkg Kernel fit  - GP bkgnd kernel fit")
        can.ax.legend(framealpha=0)
        ##
        #can.ratio.stem(xBkg, GPSigOnlySignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xBkg, mu_xBkg, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

if __name__ == '__main__':
    run_bkgnd()
