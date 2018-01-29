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

#Data=S+B
    #Data processing, cutting out the not desired range
    xSB, ySB, xerrSB, yerrSB = dataCut(xMin, xMax, 0, xRaw, yRaw, xerrRaw, yerrRaw) # for GP bkgnd kernel and signal kernel 
    #simple fit function 
    xSBFit, ySBFit, xerrSBFit, yerrSBFit= dataCut(xMinFit, xMaxFit, 0, xRaw, yRaw,xerrRaw,yerrRaw) # for fit function 
    #official fit fucntion fit 
    xSBOffFit, ySBOffFit, xerrOffFit, yerrOffFit = dataCut(xMinFit, xMaxFit, 0, xRawOffFit, yRawOffFit, xerrRawOffFit, yerrRawOffFit)

    print("ySBOfffFit:", ySBOffFit);
    # calculate the min log likelihood of the official fit function
    logLikeOff=logLike_3ffOff(xSB, ySB, xSBOffFit, ySBOffFit, xerrSB)

#Data=B
#getting the bkgnd data points
    xRawbk, yRawbk, xerrRawbk, yerrRawbk = getDataPoints(args.bkgnd_file, 'dijetgamma_g85_2j65','Zprime_mjj_var')
    xBkgbk, yBkgbk, xerrBkgbk, yerrBkgbk = dataCut(xMin, xMax, 0, xRawbk, yRawbk, xerrRawbk, yerrRawbk) # for GP bkgnd kernel and signal kernel 

#Data=S
# to be added 

# calculating the signal value 
    if np.any(np.not_equal(xSB,xBkgbk)):
        print("warning, xBkg and xBkgbk value are different")
    
    ySig = ySB-yBkgbk
    print("ySig", ySig)

    # make an evently spaced x
    t = np.linspace(np.min(xSB), np.max(xSB), 500)

    #Drawing the raw data of just the bkgnd and Sig + bkgnd
    #with Canvas(f'RawSig') as can: # just to check and see everything is okay
    #    can.ax.errorbar(xSB, ySB, yerr=yerrSB, fmt='.g')
    #    can.ax.set_yscale('log')

    # data SB. kernel: GP bkgnd: finidng the mean and covaraicne 
    ymuGP_KernBkg_SB, covGP_KernBkg_SB, bestFitSB=y_bestFitGP(xSB,ySB,xerrSB, yerrSB,3, kernelType="bkg")

    # data SB, kernel: GP Signal: finind the mean of covariance 
    ymuGP_KernBG_SB, cov_xSig, bestFitSig= y_bestFitGP(xSB, ySB, xerrSB, yerrSB, 1, kernelType="sig")

    #Signal only fit
    ySigFit= ymuGP_KernBG_SB-ymuGP_KernBkg_SB
    print("ySigFit: ",ySigFit)

#finding the fit y values 
    fit_mean=y_bestFit3Params(xSBFit, ySBFit, xerrSBFit, 1)

    kargs = {x:y for x, y in bestFitSig.items()}
    hyperParams=kargs.values()
    print("hyperParams: ",hyperParams)
# Finding significance 
    MAP_GP, MAP_sig, MAP_bkg=runGP_SplusB(ySB, xSB, xerrSB,xSBFit, xerrSBFit, hyperParams)
    print("MAP_bkg: ", MAP_bkg);
    print("MAP_sig: ", MAP_sig);


#finding signifiance 
    GPSignificance, chi2=res_significance(ySB, ymuGP_KernBkg_SB)
    fitSignificance, chi2fit=res_significance(ySBFit, fit_mean)
    GPSigSignificance, chi2SignalFit=res_significance(ySB, ymuGP_KernBG_SB)
    GPSigOnlySignificance, chi2SigOnlySignificance = res_significance(ySig, ySigFit)
    GPSigOnlySignificanceMeg, chi2SigOnlySignificanceMeg = res_significance(ySig,MAP_sig )
    FitSigFromOff, chi2FitOff = res_significance(ySB,ySBOffFit)
    print("official significance: ", ySBOffFit);
    #print("x",xSB)
    #print("significance: ",GPSigOnlySignificance)

#drawing the result
    ext = args.output_file_extension
    title="test"
    with Canvas(f'%s{ext}'%title, "Fit Function Official ", "GP bkgnd kernel", "GP signal+bkgnd kernel", 3) as can:
        can.ax.errorbar(xSB, ySB, yerr=yerrSB, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(xSBFit, fit_mean, '.r', label="fit function")
        can.ax.plot(xSBOffFit, ySBOffFit, '-m', label="fit function official")
        can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.ax.plot(xSB, ymuGP_KernBG_SB, '-b', label="GP signal kernel") 
        can.ax.legend(framealpha=0)
        can.ratio.stem(xSBOffFit, FitSigFromOff, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, fitSignificance, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, testsig, markerfmt='.', basefmt=' ')
        can.ratio.set_ylabel("significance")
        can.ratio2.stem(xSB, GPSignificance, markerfmt='.', basefmt=' ')
        can.ratio2.set_ylabel("significance")
        can.ratio3.set_ylabel("significance")
        can.ratio3.stem(xSB, GPSigSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.ratio3.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

#drawing the signal 
    ext = args.output_file_extension
    title="signalonly"
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, ySigFit, '-r', label="GP Sig + bkg Kernel fit  - GP bkgnd kernel fit")
        can.ax.legend(framealpha=0)
        ##
        can.ratio.stem(xSB, GPSigOnlySignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

#drawing the signal with meghan's method 
    ext = args.output_file_extension
    title="signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPSBFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio.stem(xSB, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

    print("min LL of off: ",logLikeOff)

#calculting with the signal kernel -
if __name__ == '__main__':
    run_sig()
