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
from Libplotmeghan import getDataPoints,dataCut, removeZeros,y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff

dataFileHistTemplate='MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1'
officialFitHistTemplate='basicBkgFrom4ParamFit'

class dataSet: # each class treats a type of data set
    """each dataSet class  makes predictions using 
        1. GP bkgnd Kernel
        2. GP bkgnd+ signal Kernel
        3. Simple Fit function
        4. Official Fit Function 
        5.SWIFT (Coming soon) """

    def __init__(self, xMinData, xMaxData, xMinSimpleFit, xMaxSimpleFit, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate):
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

        # make an evently spaced x
        t = np.linspace(np.min(self.xData), np.max(self.xData), 500) ##Is this only used in drawing?

        # boolean counter to see if cetain things are done 
        simpleFitDone=False
        gPBkgKernelFitDone=False
        gPSigPlusBkgKernelFitDone=False

  #EXECUTIOi
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


    def gPBkgKernelFit(self, trial=1, useSimpleFitResult=False):
        """Execute GPBkgKernel Fit"""
        gPBkgKernelFitDone=True 
        self.y_GPBkgKernelFit, self.cov_GPBkgKernelFit, self.bestFit_GPBkgKernelFit=y_bestFitGP(self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="bkg")
        self.significance_GPBkgKernelFit, chi2_GPBkgKernelFit=res_significance(self.yData, self.y_GPBkgKernelFit)

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

    def officialFit(self, print =False):
        """official Fit """
        #need to add a thing to check whether the initial value of the yData and y_officaiFit matches       o 
        self.yFit_officialFit=self.yOffFit
        self.significance_officialFit, self.chi2_officialFit = res_significance(self.yData,self.yOffFit)  
        if print==True:
            self.printOfficialFit()

    def printOfficialFit(self):
        print("-----------------Testing Official fit output--------------------")
        print("xFit: ", self.xFit_officialFit)
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

    
def y_SignalData(signalBkgDataSet, bkgDataSet):
    if np.any(np.not_equal(signalBkgDataSet.xData,bkgDataSet.xData)):
        print("Error, xBkg and xBkgbk value are different")
        return  1
    else :
        ySigData = signalBkgDataSet.yData-bkgDataSet.yData
        print("ySig", ySig)
        return ySigData

def y_SignalGPSubtractionFit(sigPlusBkgFit, bkgFit):
    pass

def y_signalGaussianFit
    pass




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
    
    signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5", officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
    #signalInjectedBkgndData.simpleFit()
    #signalInjectedBkgndData.printSimpleFit()
    #signalInjectedBkgndData.gPBkgKernelFit()
    #signalInjectedBkgndData.printGPBkgKernelFit()
    #signalInjectedBkgndData.gPSigPlusBkgKernelFit()
    #signalInjectedBkgndData.printGPSigPlusBkgKernelFit()
    signalInjectedBkgndData.officialFit(print=True)

    signalInjectedBkgndData.fitAll(print=True)
         
    
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    bkgndData.simpleFit()
    bkgndData.printSimpleFit()
    bkgndData.gPBkgKernelFit()
    bkgndData.printGPBkgKernelFit()
    bkgndData.gPSigPlusBkgKernelFit()
    bkgndData.printGPSigPlusBkgKernelFit()

    ySignalData = y_signalData(signalInjectedBkgndData, bkgndData)
    print("ySignalData: ", ySignalData)

## drawing stuff



##    def __init__(self, xMinData, xMaxData, xMinSimpleFit, xMaxSimpleFit, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate):
