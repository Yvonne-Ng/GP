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
import random

#importing functions from Libplotmeghan
from Libplotmeghan import getDataPoints,dataCut, removeZeros,makeToys,y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff, gaussian
from classData import *
#import function from drawStuff
from drawStuff import drawSignalGaussianFit, drawSignalSubtractionFit, drawFitDataSet, drawAllSignalFit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#for the gaussian fit 
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import defaultdict

class toyDataSetCollection: #for ease of holding dataSet collection like a bunch of toys
    def __init__(self, dataSetOriginal, nToys=1 , fitAll=True,fitTrial=None):
    #-----Making toy list
        self.toyList=[]
        for i in range(nToys):
            self.toyList.append(dataSet(toy=True, originalSet= dataSetOriginal)) 
    #----Making dicts for other variables
        self.dataSetOriginal=dataSetOriginal
        self.nToys=nToys
        if fitTrial==None:
            self.fitTrial=1 #default use the fit Trial # of the original data set
        else:
            self.fitTrial=fitTrial
    #---Making default dict
        self.chi2List= defaultdict(list)
        self.logLikelihood={}
        self.yFittedList={}
    #----Executing the fit ##default is to fit all of the toyDataSet
        if fitAll:
            self.fitAll(self.fitTrial)


    def fitAll(self, fitrial=5):
        #use all the initial varriables as the dataSetCollection values
        for i in range(self.nToys):
            self.toyList[i].simpleFit(trial=self.fitTrial,doPrint=False)
            self.toyList[i].gPBkgKernelFit(trial=self.fitTrial,bkgDataHyperParams=self.dataSetOriginal.bkgData_GPBkgParams, doPrint=False)
            self.toyList[i].gPSigPlusBkgKernelFit(trial=self.fitTrial, doPrint=False)
            #don't really need to reconstruct the signal 
            #self.toyList[i].gyGPSignalReconstructed_dataSBMinusB(doPrint=False)
            #self.toyList[i].officialFit(doPrint=True)


    def GetChi2List(self, fit="simpleFit"): #fit can be @simpleFit @GPBkg @GPBkgSig
    #TODO add in error catch exception if it's not the right kind of fit
        for dataSet in self.toyList:
            print(dataSet.chi2[fit])
            self.chi2List[fit].append(dataSet.chi2[fit])
        return self.chi2List[fit]

if __name__=="__main__":        
    pass
    
    
     
    
