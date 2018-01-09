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
from Libplotmeghan import getDataPoints,dataCut, removeZeros,y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff, gauss
import random

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

dataFileHistTemplate='MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1'
officialFitHistTemplate='basicBkgFrom4ParamFit'

class dataSet: # each class treats a type of data set
    """each dataSet class  makes predictions using 
        1. GP bkgnd Kernel
        2. GP bkgnd+ signal Kernel
        3. Simple Fit function
        4. Official Fit Function 
        5.SWIFT (Coming soon) """

    def __init__(self, xMinData=None, xMaxData=None, xMinSimpleFit=None, xMaxSimpleFit=None, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate, toy=False, originalSet=None):
        if toy=False:
        #Getting Data points
            self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw = getDataPoints(dataFile, dataFileDir, dataFileHist)
            self.xRawOffFit, self.yRawOffFit, self.xerrRawOffFit, self.yerrRawOffFit = getDataPoints(officialFitFile, officialFitDir, officialFitHist)
        #Cutting out the desired range 
            #Data file (For GP Fitting)
            self.xData, self.yData, self.xerrData, self.yerrData = dataCut(xMinData, xMaxData, 0, self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw)  
            #Simple fit range 
            self.x_simpleFit, self.y_simpleFit, self.xerr_simpleFit, self.yerr_simpleFit= dataCut(xMinSimpleFit, xMaxSimpleFit, 0, self.xRaw, self.yRaw,self.xerrRaw,self.yerrRaw) # for fit function 
            #Official Fit (Already cut out in root file )
            #(self.xOffFit, self.yOffFit, self.xerrOffFit, self.yerrOffFit)=(self.xRawOffFit, self.yRawOffFit, self.xerrRawOffFit, self.yerrRawOffFit)
            self.xOffFit=self.xRawOffFit
            self.yOffFit=self.yRawOffFit
            self.xerrOffFit = self.xerrRawOffFit
            self.yerrOffFit = self.yerrRawOffFit
            #removing the zeros
            self.xOffFit, self.yOffFit, self.xerrOffFit, self.yerrOffFit=removeZeros(self.xOffFit, self.yOffFit, self.xerrOffFit, self.yerrOffFit)

        else: # toy=True
            self.xData, self.x_simpleFit= originalSet.xData
            self.yData, self.y_simpleFit= originalSet.yData
            self.xerrData, self.xerr_simpleFit = originalSet.xerrData
            self.yerrData, self.yerr_simpleFit = originalSet.yerrData
            #no official fit for toys #add from 

        # make an evently spaced x
        t = np.linspace(np.min(self.xData), np.max(self.xData), 500) ##Is this only used in drawing?

        # boolean counter to see if cetain things are done 
        simpleFitDone=False
        gPBkgKernelFitDone=False
        gPSigPlusBkgKernelFitDone=False
        
  # Getting stuff
    def get_simpleFit_chi2(self):
        return self.chi2_simpleFit

    def get_GPBkgKernelFit_chi2(self):
        return self.chi2_GPBkgKernelFit

        
  #EXECUTION
    def simpleFit(self, trial=1):
        """Execute simple fit """
        simpleFitDone=True
        self.yFit_simpleFit=y_bestFit3Params(self.x_simpleFit, self.y_simpleFit, self.xerr_simpleFit, trial)
        ### add parameter
        self.significance_simpleFit, self.chi2_simpleFit=res_significance(self.y_simpleFit, self.yFit_simpleFit)
        

    def printSimpleFit(self):
        print("-------------testing Simple Fit------------")
        print("yFit: ", self.yFit_simpleFit)
        print("significance: ", self.significance_simpleFit)


    def gPBkgKernelFit(self, trial=1, useBkgKernelResult="", useSimpleFitResult=False): 
        """Execute GPBkgKernel Fit"""
        gPBkgKernelFitDone=True 
        self.y_GPBkgKernelFit, self.cov_GPBkgKernelFit, self.bestFit_GPBkgKernelFit=y_bestFitGP(self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="bkg")
        self.significance_GPBkgKernelFit, self.chi2_GPBkgKernelFit=res_significance(self.yData, self.y_GPBkgKernelFit)

    def printGPBkgKernelFit(self):
        print("-------------testing GP bkgnd kernel Fit------------")
        print("yFit: ", self.y_GPBkgKernelFit)
        print("significance: ", self.significance_GPBkgKernelFit)

    def gPSigPlusBkgKernelFit(self, trial=1):
        gPSigPlusBkgKernelFitDone=True
        """Execute GP Signal plus bkg fit """
        self.y_GPSigPlusBkgKernelFit, self.cov_GPSigPlusBkgKernelFit, self.bestFit_GPSigPlusBkgKernelFit=y_bestFitGP(self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="sig")
        self.significance_GPSigPlusBkgKernelFit, chi2=GPSigPlusBkgKernelFit=res_significance(self.yData, self.y_GPSigPlusBkgKernelFit)
        
    def printGPSigPlusBkgKernelFit(self):
        print("-------------testing GP signal plus bkgnd kernel fit ------------")
        print("yFit: ", self.y_GPSigPlusBkgKernelFit)
        print("significance: ", self.significance_GPSigPlusBkgKernelFit)

    def officialFit(self, doprint=False):
        """official Fit """
        #need to add a thing to check whether the initial value of the yData and y_officaiFit matches       o 
        self.yFit_officialFit=self.yOffFit
        self.significance_officialFit, self.chi2_officialFit = res_significance(self.yData,self.yOffFit)  
        if doprint==True:
            self.printOfficialFit()

    def printOfficialFit(self):
        print("-----------------Testing Official fit output--------------------")
        print("xFit: ", self.xOffFit)
        print("xData: ", self.xData)
        print("yFit: ", self.yFit_officialFit)
        print("significance", self.significance_officialFit)

    def fitAll(self, print=False):
        self.simpleFit()
        self.gPBkgKernelFit()
        self.gPSigPlusBkgKernelFit()
        self.officialFit()
        if (print==True):
            self.printSimpleFit()
            self.printGPBkgKernelFit()
            self.printGPSigPlusBkgKernelFit()
            self.printOfficialFit()

#class signalDataSet():
#    def__init__(s
    
def y_signalData(signalBkgDataSet, bkgDataSet):
    if np.any(np.not_equal(signalBkgDataSet.xData,bkgDataSet.xData)):
        print("Error, xBkg and xBkgbk value are different")
        return  1
    else :
        ySigData = signalBkgDataSet.yData-bkgDataSet.yData
        print("xData", signalBkgDataSet.xData)
        print("xSignal", signalBkgDataSet.xOffFit)
        return ySigData

def y_signalGPSubtractionFit(signalBkgDataSet, doPrint=False):
    Fit =sigBkgDataSet.y_GPSigPlusBkgKernelFit- signalBkgDataSet.y_GPBkgKernelFit
    if doPrint:
        print (" y_signalalGPSubtractionFit: ", Fit)
    return 

def y_signalGaussianFit(signalBkgDataSet, ySignalData, doPrint):
    p_initial = [1.0, 0.0, 0.1, 0.0]
    popt, pcov = curve_fit(gauss, signalBkgDataSet.xData, ySignalData, p0=p_initial, sigma=signalBkgDataSet.yerrData)
    Fit=gauss(xData, *popt)
    if doPrint:
        print("y signal Gaussian Fit: ", Fit)
    return Fit


#making toys
    def makeListofToySet(dataSetOriginal, nToy=1):
        toyDataSetList=[]
        for i in range(nToys):
            toyDataSetList.append(dataSet(toy=True, dataSetOriginal)) 
        return toyDataSetList

def yGPSignalReconstructed_dataSBMinusB(bkgndData):
    pass
        
if __name__=="__main__":        
    print("----------------------")
    print("----------------------")
    print("----------------------")
    print("------test---------")
    print("----------------------")
    print("----------------------")
    print("----------------------")
    
#signal plus bkgnd Data
    signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5", officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
    signalInjectedBkgndData.fitAll(print=True)

#making a list of toys for the signal plus bkgnd Data 
    signalInjectedBkgndToy=makeToyList(signalInjectedBkgndData, 100)

         
#signal plus bkgnd Data
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    bkgndData.fitAll(print=True)

##signalData
    ySignalData = y_signalData(signalInjectedBkgndData, bkgndData)
    ySignalGPSubtractionFit=y_signalGPSubtractionFit(signalInjectedBkgndData, doPrint=True)
    ySignalGaussianFit=y_signalGaussianFit(doPrint=True)
    

## drawing stuff



##    def __init__(self, xMinData, xMaxData, xMinSimpleFit, xMaxSimpleFit, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate):
