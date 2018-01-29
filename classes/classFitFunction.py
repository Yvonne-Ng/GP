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


    def grabNProcessData(self,xMin, xMax, xRaw, yRaw, xerrRaw, yerrRaw):
        self.grabbedData=True
        self.xData, self.yData, self.xerrData, self.yerrData=dataCut(xMin, xMax, 0., xRaw, yRaw,xerrRaw, yerrRaw)
    
    #-------Running the Fit 
    def doFit(self, initFitParam=None, initRange=None, trial=100):
        self.fitParam=initFitParam
        self.rangeFitParam=initRange
        self.trial=trial

        if self.grabbedData==True:
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
        lnProbUA2 = logLike_UA2(self.xData,self.yData,self.xerrData)
        #where the initRange is set
        minimumLLH, best_fit_params = fit_UA2(self.trial,  lnProbUA2,initParam=self.fitParam, initRange=self.rangeFitParam)
        fit_mean = model_UA2(self.xData, best_fit_params, self.xerrData)
        self.bestFitParams=best_fit_params
        return fit_mean 

    def std4ParamsFit(self):
        #----4 param fit function
        lnProb = logLike_4ff(self.xData,self.yData,self.xerrData)
        minimumLLH, best_fit_params = fit_4ff(self.trial, lnProb,initParam=self.fitParam, initRange=self.rangeFitParam)
        fit_mean = model_4param(self.xData, best_fit_params, self.xerrData)
        self.bestFitParams=best_fit_params
        return fit_mean
       
if __name__=="__main__":
    #Testing the class
    #
    #-----create a dataSet
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/dijetgamma_g85_2j65_nbtag2.h5",dataFileDir="", dataFileHist="background_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    # Create a fit function 
    UAFitBkgndMC=FitFunction(0)
    UAFitBkgndMC.grabNProcessData(bkgndData.xMinData, bkgndData.xMaxData, bkgndData.xData, bkgndData.yData, bkgndData.xerrData, bkgndData.yerrData)

    yFit=UAFitBkgndMC.doFit()
    print(yFit)
    sig=res_significance(yFit,bkgndData.yData)
    drawFit(bkgndData.xData, yFit, bkgndData.yData, sig,"btagged2")

    #print(UAFitBkgndMC.doFit(initFitParam=(10,-2, 56.87,-75.87),initRange=[(0, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)] ))


