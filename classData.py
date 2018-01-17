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
#import function from drawStuff
from drawStuff import drawSignalGaussianFit, drawSignalSubtractionFit, drawFitDataSet, drawAllSignalFit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#for the gaussian fit 
from scipy.stats import norm
import matplotlib.pyplot as plt

dataFileHistTemplate='MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1'
officialFitHistTemplate='basicBkgFrom4ParamFit'
FIT3_PARS = ['p0','p1','p2']

class dataSet: # each class treats a type of data set
    """each dataSet class  makes predictions using 
        1. GP bkgnd Kernel
        2. GP bkgnd+ signal Kernel
        3. Simple Fit function
        4. Official Fit Function 
        5.SWIFT (Coming soon) """

    def __init__(self, xMinData=None, xMaxData=None, xMinSimpleFit=None, xMaxSimpleFit=None, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate, toy=False, originalSet=None):
        if toy==False:
        #Getting Data points
            self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw = getDataPoints(dataFile, dataFileDir, dataFileHist)
            print("print xDataRaw:", self.xRaw)
            self.xRawOffFit, self.yRawOffFit, self.xerrRawOffFit, self.yerrRawOffFit = getDataPoints(officialFitFile, officialFitDir, officialFitHist)
        #Cutting out the desired range 
            #Data file (For GP Fitting)
            self.xData, self.yData, self.xerrData, self.yerrData = dataCut(xMinData, xMaxData, 0, self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw)  
            print("print xData:", self.xData)
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
            #

        else: # toy=True
            self.xData=self.x_simpleFit= makeToys(originalSet.xData, lumi=34.5)
            self.yData=self.y_simpleFit= makeToys(originalSet.yData, lumi=34.5)
            self.xerrData=self.xerr_simpleFit = makeToys(originalSet.xerrData, lumi=34.5)
            self.yerrData=self.yerr_simpleFit = makeToys(originalSet.yerrData, lumi=34.5)
            #no official fit for toys #add from 

        # make an evently spaced x
        self.x_gPPred = np.linspace(np.min(self.xData), np.max(self.xData), 500) ## this is for x_GPpred

        self.xerr_gPPred=self.x_gPPred[1:]-self.x_gPPred[:-1]
        print("size of xerr_gPPred: ",np.size(self.xerr_gPPred))
        self.xerr_gPPred=np.append(self.xerr_gPPred, self.xerr_gPPred[-1])
        print("size of xerr_gPPred: ",np.size(self.xerr_gPPred))


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
    def simpleFit(self, trial=1, doPrint=False):
        """Execute simple fit """
        simpleFitDone=True
        self.yFit_simpleFit=y_bestFit3Params(self.x_simpleFit, self.y_simpleFit, self.xerr_simpleFit, trial)
        ### add parameter
        self.significance_simpleFit, self.chi2_simpleFit=res_significance(self.y_simpleFit, self.yFit_simpleFit)
        if doPrint:
            print("-------------testing Simple Fit------------")
            print("yFit: ", self.yFit_simpleFit)
            print("significance: ", self.significance_simpleFit)

    def gPBkgKernelFit(self, trial=100, useBkgDataResult="", useSimpleFitResult=False,bkgDataHyperParams=None, doPrint=False): 
        """Execute GPBkgKernel Fit"""
        gPBkgKernelFitDone=True 
        self.y_GPBkgKernelFit, self.cov_GPBkgKernelFit, self.bestFit_GPBkgKernelFit=y_bestFitGP(self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="bkg", bkgDataParams=None) #bkgDataHyperParam will be None unless set when the function is called 
        self.significance_GPBkgKernelFit, self.chi2_GPBkgKernelFit=res_significance(self.yData, self.y_GPBkgKernelFit)
        if doPrint:
            print("-------------testing GP bkgnd kernel Fit------------")
            print("yFit: ", self.y_GPBkgKernelFit)
            print("significance: ", self.significance_GPBkgKernelFit)

# y TODO is this ever used?
    def getGPBkgKernelFitParams(self):
        return self.bestFit_GPBkgKernelFit

    def gPSigPlusBkgKernelFit(self, trial=100, doPrint=False):
        gPSigPlusBkgKernelFitDone=True
        """Execute GP Signal plus bkg fit """
        self.y_GPSigPlusBkgKernelFit, self.cov_GPSigPlusBkgKernelFit, self.bestFit_GPSigPlusBkgKernelFit=y_bestFitGP(self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="sig")
        self.significance_GPSigPlusBkgKernelFit, chi2=GPSigPlusBkgKernelFit=res_significance(self.yData, self.y_GPSigPlusBkgKernelFit)
        if doPrint:
            print("-------------testing GP signal plus bkgnd kernel fit ------------")
            print("yFit: ", self.y_GPSigPlusBkgKernelFit)
            print("significance: ", self.significance_GPSigPlusBkgKernelFit)
        
    def yGPSignalReconstructed_dataSBMinusB(self, doPrint=False):
        print("self.bestFit_GPSigPlusBkgKernelFit", self.bestFit_GPSigPlusBkgKernelFit)
        self.MAP_GP, self.MAP_sig, self.MAP_bkg=runGP_SplusB(self.yData, self.xData, self.xerrData,self.x_gPPred, self.xerr_gPPred, self.bestFit_GPSigPlusBkgKernelFit.values())
        if doPrint:
            print("MAP_GP: ", self.MAP_GP)
            print("MAP_sig: ", self.MAP_sig)
            print("MAP_bkg: ", self.MAP_bkg)

    def officialFit(self, doPrint=False):
        """official Fit """
        #need to add a thing to check whether the initial value of the yData and y_officaiFit matches       o 
        self.yFit_officialFit=self.yOffFit
        self.significance_officialFit, self.chi2_officialFit = res_significance(self.yData,self.yOffFit)  
        if doPrint==True:
            print("-----------------Testing Official fit output--------------------")
            print("xFit: ", self.xOffFit)
            print("xData: ", self.xData)
            print("yFit: ", self.yFit_officialFit)
            print("significance", self.significance_officialFit)


    def fitAll(self, trialAll=100,bkgDataParams=None):
        self.simpleFit(trial=trialAll,doPrint=True)
        self.gPBkgKernelFit(trial=trialAll,bkgDataHyperParams=bkgDataParams, doPrint=True)
        self.gPSigPlusBkgKernelFit(trial=trialAll, doPrint=True)
        self.yGPSignalReconstructed_dataSBMinusB(doPrint=True)
        self.officialFit(doPrint=True)

class signalDataSet():
    def __init__(self,signalBkgDataSet, bkgDataSet): 
        #make sure the size of both data matches 
        # finding the raw data points
        if np.any(np.not_equal(signalBkgDataSet.xData,bkgDataSet.xData)):
            print("Error, xBkg and xBkgbk value are different")
            return  1
        else :
            self.ySigData = signalBkgDataSet.yData-bkgDataSet.yData
            self.xSigData = signalBkgDataSet.xData
        #GP subtraction fit 
        self.yGPSubtractionFit =signalBkgDataSet.y_GPSigPlusBkgKernelFit- signalBkgDataSet.y_GPBkgKernelFit
        self.gpSubtractionSignificance=res_significance(self.ySigData, self.yGPSubtractionFit)
        #need to have a very good initial guess in order to have the gaussian converge
        peakValue=self.ySigData.max()
        mean = self.xSigData[self.ySigData.argmax()]
        sigma = mean - np.where(self.ySigData > peakValue * np.exp(-.5))[0][0] 
        init_vals = [peakValue, mean, sigma] 
        best_vals, covar = curve_fit(gaussian, self.xSigData, self.ySigData, p0=init_vals)
        self.yGaussianFit = gaussian(signalBkgDataSet.xData, *best_vals)
        self.gaussianFitSignificance=res_significance(self.ySigData, self.yGaussianFit)
        #print best_vals

    def print(self):
        print("xSigData: ",self.xSigData)
        print("ySigData: ",self.ySigData)
        print ("GP subtraction fit y : ", self.yGPSubtractionFit )
        print ("GP siginificaance : ", self.gpSubtractionSignificance)
        print("Gaussian Fit : ", self.yGaussianFit)
        print ("Gaussian significance: ", self.gaussianFitSignificance)

        
        # Reconstruction 
        #fit range needs to be different from train range 
        #self.runGP_SplusB(self.ySigData, self.xSigData, self.xerr

#def y_signalData(signalBkgDataSet, bkgDataSet):
#
#
#def y_signalGPSubtractionFit(signalBkgDataSet, doPrint=False):
#    Fit =sigBkgDataSet.y_GPSigPlusBkgKernelFit- signalBkgDataSet.y_GPBkgKernelFit
#    if doPrint:
#        print (" y_signalalGPSubtractionFit: ", Fit)
#    return 
#
#def y_signalGaussianFit(signalBkgDataSet, ySignalData, doPrint):
#    p_initial = [1.0, 0.0, 0.1, 0.0]
#    popt, pcov = curve_fit(gauss, signalBkgDataSet.xData, ySignalData, p0=p_initial, sigma=signalBkgDataSet.yerrData)
#    Fit=gauss(xData, *popt)
#    if doPrint:
#        print("y signal Gaussian Fit: ", Fit)
#    return Fit

#making toys
def makeToyDataSetList(dataSetOriginal, nToy=1):
    toyDataSetList=[]
    for i in range(nToy):
        toyDataSetList.append(dataSet(toy=True, originalSet= dataSetOriginal)) 
    return toyDataSetList

#def yGPSignalReconstructed_dataSBMinusB(dataSet):
#    MAP_GP, MAP_sig, MAP_bkg=runGP_SplusB(dataSet.yData, data.xData, self.xerrData,self.x_gPPred, self.xerr_gPPred, hyperParams)
#    
        
if __name__=="__main__":        
    print("----------------------")
    print("----------------------")
    print("----------------------")
    print("------test---------")
    print("----------------------")
    print("----------------------")
    print("----------------------")
    
#signal plus bkgnd Data
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    bkgndData.fitAll( trialAll=1)

#signal plus bkgnd Data
#    signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5", officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
#    signalInjectedBkgndData.fitAll(print=True, trialAll=1, bkgDataParams=bkgndData.getGPBkgKernelFitParams())
#
##making a list of toys for the signal plus bkgnd Data 
#    signalInjectedBkgndToy=makeToyDataSetList(signalInjectedBkgndData, 1)
#
#         
###signalData
##    ySignalData = y_signalData(signalInjectedBkgndData, bkgndData)
##    ySignalGPSubtractionFit=y_signalGPSubtractionFit(signalInjectedBkgndData, doPrint=True)
##    ySignalGaussianFit=y_signalGaussianFit(doPrint=True)
#    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
#    signalData1.print()
###test ToyList
### drawing stuff
#
#    #drawSignalGaussianFit(signalInjectedBkgndData, signalData1)
#    drawAllSignalFit(signalInjectedBkgndData, signalData1)
#    drawFitDataSet(signalInjectedBkgndData, "TestSignalinjectedBkg")
    drawFitDataSet(bkgndData, "Bkg", saveTxt=True, saveTxtDir="txt/BkgData")

#
#
###    def __init__(self, xMinData, xMaxData, xMinSimpleFit, xMaxSimpleFit, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate):
