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
from Libplotmeghan import getDataPoints,dataCut, y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('signal_plus_bkgndFile')
    parser.add_argument('bkgnd_file')
    parser.add_argument('signalplusbkgndOffFit')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    return parser.parse_args()


def run_sig():
    xMin=300
    xMax=1500
    xMinFit=300
    xMaxFit=1500

    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.signal_plus_bkgndFile, '', 'MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1')
    xRawOffFit, yRawOffFit, xerrRawOffFit, yerrRawOffFit = getDataPoints(args.signalplusbkgndOffFit, '', 'basicBkgFrom4ParamFit')

    #Data processing, cutting out the not desired range
    xBkg, yBkg, xerrBkg, yerrBkg = dataCut(xMin, xMax, 0, xRaw, yRaw, xerrRaw, yerrRaw) # for GP bkgnd kernel and signal kernel 
    xBkgFit, yBkgFit, xerrBkgFit, yerrBkgFit= dataCut(xMinFit, xMaxFit, 0, xRaw, yRaw,xerrRaw,yerrRaw) # for fit function 
    #official fit fucntion fit
    xFit, yFit, xerrFit, yerrFit = dataCut(xMinFit, xMaxFit, 0, xRawOffFit, yRawOffFit, xerrRawOffFit, yerrRawOffFit)

    # calculate the min log likelihood of the official fit function
    logLikeOff=logLike_3ffOff(xBkg, yBkg, xFit, yFit, xerrBkg)

#getting the bkgnd data points
    xRawbk, yRawbk, xerrRawbk, yerrRawbk = getDataPoints(args.bkgnd_file, 'dijetgamma_g85_2j65','Zprime_mjj_var')
    xBkgbk, yBkgbk, xerrBkgbk, yerrBkgbk = dataCut(xMin, xMax, 0, xRawbk, yRawbk, xerrRawbk, yerrRawbk) # for GP bkgnd kernel and signal kernel 
# calculating the signal value 
    if np.any(np.not_equal(xBkg,xBkgbk)):
        print("warning, xBkg and xBkgbk value are different")
    
    ySig = yBkg-yBkgbk
    print("ySig", ySig)

    # make an evently spaced x
    t = np.linspace(np.min(xBkg), np.max(xBkg), 500)

    #Drawing the raw data of just the bkgnd and Sig + bkgnd
    #with Canvas(f'RawSig') as can: # just to check and see everything is okay
    #    can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g')
    #    can.ax.set_yscale('log')

    #GP bkgnd: finidng the mean and covaraicne 
    mu_xBkg, cov_xBkg, bestFitBkg=y_bestFitGP(xBkg,yBkg,xerrBkg, yerrBkg,3, kernelType="bkg")

    # GP Signal: finind the mean of covariance 
    mu_xSig, cov_xSig, bestFitSig= y_bestFitGP(xBkg, yBkg, xerrBkg, yerrBkg, 100, kernelType="sig")

    #Signal only fit
    ySigFit= mu_xSig-mu_xBkg
    print("ySigFit: ",ySigFit)

#finding the fit y values 
    fit_mean=y_bestFit3Params(xBkgFit, yBkgFit, xerrBkgFit, 1)

    kargs = {x:y for x, y in bestFitSig.items()}
    hyperParams=kargs.values()
    print("hyperParams: ",hyperParams)
# Finding significance 
    MAP_GP, MAP_sig, MAP_bkg=runGP_SplusB(yBkg, xBkg, xerrBkg,xBkgFit, xerrBkgFit, hyperParams)
      #  MAP_GP, MAP_bkg, MAP_sig = runGP_SplusB(ydata, xtoy, xtoyerr, xbins, xerrs, hyperParams)


#finding signifiance 
    GPSignificance, chi2=res_significance(yBkg, mu_xBkg)
    fitSignificance, chi2fit=res_significance(yBkgFit, fit_mean)
    GPSigSignificance, chi2SignalFit=res_significance(yBkg, mu_xSig)
    GPSigOnlySignificance, chi2SigOnlySignificance = res_significance(ySig, ySigFit)
    GPSigOnlySignificanceMeg, chi2SigOnlySignificanceMeg = res_significance(ySig,MAP_sig )
    FitSigFromOff, chi2FitOff = res_significance(yBkgFit,yFit)
    print("x",xBkg)
    print("significance: ",GPSigOnlySignificance)

#drawing the result
    ext = args.output_file_extension
    title="test"
    with Canvas(f'%s{ext}'%title, "Fit Function Official ", "GP bkgnd kernel", "GP signal+bkgnd kernel", 3) as can:
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(xBkgFit, fit_mean, '-r', label="fit function")
        can.ax.plot(xFit, yFit, '-m', label="fit function official")
        can.ax.plot(xBkg, mu_xBkg, '-g', label="GP bkgnd kernel") #drawing 
        can.ax.plot(xBkg, mu_xSig, '-b', label="GP signal kernel") 
        can.ax.legend(framealpha=0)
        can.ratio.stem(xFit, FitSigFromOff, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xBkgFit, fitSignificance, markerfmt='.', basefmt=' ')
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

#drawing the signal 
    ext = args.output_file_extension
    title="signalonly"
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        can.ax.errorbar(xBkg, ySig, yerr=yerrBkg, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        can.ax.set_yscale('log')
        can.ax.plot(xBkg, ySigFit, '-r', label="GP Sig + bkg Kernel fit  - GP bkgnd kernel fit")
        can.ax.legend(framealpha=0)
        ##
        can.ratio.stem(xBkg, GPSigOnlySignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xBkg, mu_xBkg, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

#drawing the signal with meghan's method 
    ext = args.output_file_extension
    title="signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        can.ax.errorbar(xBkg, ySig, yerr=yerrBkg, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xBkg, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio.stem(xBkg, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xBkg, mu_xBkg, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

    print("min LL of off: ",logLikeOff)

#calculting with the signal kernel -
if __name__ == '__main__':
    run_sig()
