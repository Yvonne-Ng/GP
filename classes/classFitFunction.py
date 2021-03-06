#!/usr/bin/env python3


from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from iminuit import Minuit
import scipy.special as ssp
import inspect
#from lib-plotmeghan import *
from Libplotmeghan import *
from classData import *
from drawStuff import drawFit

#---------------
# How to use this Class
# 1. Initialization
#  E.g.: fit1=FitFunction(initFunctionChoice=0, initFitParam=(0,0,0,0), initRange=[(0,0), (0,0), (0,0), (0,0),(0,0)], trial=100)

# 2. grab data and process it
#  E.g: fit1.grabNProcessData(xMin, yMin, xData, yData, xerr, yerr)

# 3. Fitting



#---------------


#---------
#function code and the corresponding functions

class FitFunction():
    def __init__(self, initFunctionChoice=0):
        self.functionChoice=initFunctionChoice

    #-------counters
        self.grabbedData=False
        self.doneFit=False


    def grabNProcessData(self,xMin, xMax, xRaw, yRaw, xerrRaw, yerrRaw,weight=None,useScaled=False):
        self.grabbedData=True
        self.xData, self.yData, self.xerrData, self.yerrData=dataCut(xMin, xMax, 0., xRaw, yRaw,xerrRaw, yerrRaw)
        if weight is not None:
            self.weight=weight

    #-------Running the Fit
    def doFit(self, initParam=None, initFitParam=None, initRange=None, trial=100,useScaled=False):
        self.initParam=initParam
        self.initFitParam=initFitParam
        self.rangeFitParam=initRange
        self.trial=trial
        self.useScaled=useScaled

        if self.grabbedData==True:
            if self.functionChoice==0:
                return self.UAFitFunction()
            if self.functionChoice==1:
                return self.std4ParamsFit()
            if self.functionChoice==2:
                return self.Param3Fit()
            self.doneFit=True
        else:
            print("can't doFit, you haven't grabbed data yet!")

    #-----different choices of fitFunctions
    def UAFitFunction(self):
        #----3 param fit function in a different way
        if self.useScaled==False:
            lnProbUA2 = logLike_UA2(self.xData,self.yData,self.xerrData, weight=self.weight)
            #where the initRange is set
        else:
            lnProbUA2 = logLike_UA2(self.xData,self.yData,self.xerrData)
            #where the initRange is set
        minimumLLH, best_fit_params = fit_UA2(self.trial,  lnProbUA2,initParam=self.initParam, initFitParam=self.initFitParam, initRange=self.rangeFitParam)
        fit_mean = model_UA2(self.xData, best_fit_params, self.xerrData)
        self.bestFitParams=best_fit_params
        return fit_mean

    def Param3Fit(self):

        if self.useScaled==False:
            lnProb=logLike_3ff(self.xData, self.yData, self.xerrData, weight =self.weight)
        else:
            lnProb=logLike_3ff(self.xData, self.yData, self.xerrData)
        minimumLLH, best_fit_params = fit_3ff(self.trial,  lnProb,initParam=self.initParam,initFitParam=self.initFitParam, initRange=self.rangeFitParam)
        fit_mean=model_3param(best_fit_params, self.xData, self.xerrData)
        print("fit mean: ", fit_mean)
        return fit_mean

    def std4ParamsFit(self):
        #----4 param fit function
        lnProb = logLike_4ff(self.xData,self.yData,self.xerrData)
        minimumLLH, best_fit_params = fit_4ff(self.trial, lnProb,initParam=self.initParam, initFitParam=self.initFitParam, initRange=self.rangeFitParam)
        fit_mean = model_4param(self.xData, best_fit_params, self.xerrData)
        self.bestFitParams=best_fit_params
        return fit_mean

if __name__=="__main__":
    #Testing the class
    #
    #-----create a dataSet
    #bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/btagged2Rebinned2.h5",dataFileDir="", dataFileHist="background_mjj_var",officialFitFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    #bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/dijetgamma_g85_2j65_nbtag1.h5",dataFileDir="", dataFileHist="background_mjj_var",officialFitFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    # Create a fit function
    useScaled=False
    if not useScaled:
        bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/dijetgamma_g85_2j65_nbtag1.h5",dataFileDir="", dataFileHist="background_mjj_var",officialFitFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
        UAFitBkgndMC=FitFunction(0)
        UAFitBkgndMC.grabNProcessData(bkgndData.xMinData, bkgndData.xMaxData, bkgndData.xData, bkgndData.yData, bkgndData.xerrData, bkgndData.yerrData,bkgndData.weighted)
        yFit=UAFitBkgndMC.doFit(trial=1, useScaled=False)
        sig, chi2=resSigYvonne(yFit,bkgndData.yData, bkgndData.weighted)
        print("yFitSize: ", yFit.shape)
        print("bkgndData.weighted size: ", bkgndData.weighted.shape)

    if useScaled:
        bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/trijet_HLT_j380_inclusive.h5",dataFileDir="", dataFileHist="background_mjj_var",officialFitFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",useScaled=True)
        UAFitBkgndMC=FitFunction(0)
        UAFitBkgndMC.grabNProcessData(bkgndData.xMinData, bkgndData.xMaxData, bkgndData.xData, bkgndData.yData, bkgndData.xerrData, bkgndData.yerrData, useScaled=True)
        yFit=UAFitBkgndMC.doFit(trial=500, useScaled=True)
        sig, chi2=resSigYvonne(yFit,bkgndData.yData, None)

    print(yFit)
    print(sig)
    drawFit(bkgndData.xData, bkgndData.yerrData, bkgndData.yData, yFit, sig,"btagged1")

    #print(UAFitBkgndMC.doFit(initFitParam=(10,-2, 56.87,-75.87),initRange=[(0, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)] ))


