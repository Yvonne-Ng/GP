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
from classData import dataSet

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
    def __init__(self, initFunctionChoice=0, initFitParam=(0,0,0,0), initRange=[(0,0), (0,0), (0,0), (0,0),(0,0)], trial=100):
        self.functionChoice=initFunctionChoice
        self.fitParams=initFitParam
        self.rangeFitParams=initRange
        self.trial=trial

    #-------counters
        self.grabbedData=False
        self.doneFit=False


    def grabNProcessData(self,xMin, xMax, xRaw, yRaw, xerrRaw, yerrRaw):
        grabbedData=True
        self.xData, self.yData, self.xerrData, self.yerrData=getDataPoints(xMin, xMax, xRaw, yRaw, xerrRaw, yerrRaw)
    
    #-------Running the Fit 
    def doFit(self):
        if grabbedData==True:
            if self.functionChoice==0:
                return self.UAFitFunction()
            if self.functionChoice==1:
                self.std4ParamsFit()
            self.doneFit=True
        else:
            print("can't doFit, you haven't grabbed data yet!")

    #-----different choices of fitFunctions
    def UAFitFunction(self):
        #----3 param fit function in a different way
        lnProbUA2 = logLike_UA2(self.xData,self.yData,self.xErrData)
        minimumLLH, best_fit_params = fit_UA2(trial, lnProbUA2)
        fit_mean = model_UA2(self.xData, best_fit_params, self.xErrData)
        self.bestFitParams=best_fit_params
        return fit_mean 

    def std4ParamsFit(self):
        #----4 param fit function
        lnProb = logLike_4ff(self.xData,self.yData,self.xErrData)
        minimumLLH, best_fit_params = fit_4ff(self.trial, lnProb)
        fit_mean = model_4param(self.xData, best_fit_params, self.xErrData)
        self.bestFitParams=best_fit_params
        return fit_mean
       
if __name__=="__main__":
    #Testing the class
    #
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    UAFitBkgndMC=FitFunction(0, trial=100)
    #def grabNProcessData(self,xMin, xMax, xRaw, yRaw, xerrRaw, yerrRaw):
    UAFitBkgndMC.grabNProcessData(bkgndData.xMinData, bkgndData.xMaxData, bkgnd.xData, bkgnd.yData, bkgnd.xerrData, yxerrData)
    print(UAFitBkgndMC.doFit())
#    def __init__(self, initFunctionChoice=0, initFitParam=(0,0,0,0), initRange[(0,0), (0,0), (0,0), (0,0)(0,0)], trial=100):


        
