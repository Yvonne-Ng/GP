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
from classDataSetCollection import *
from Libplotmeghan import getDataPoints,dataCut, removeZeros,makeToys,y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff, gaussian, resSigYvonne
#import function from drawStuff
from drawStuff import drawSignalGaussianFit, drawSignalSubtractionFit, drawFitDataSet, drawAllSignalFit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#for the gaussian fit 
from scipy.stats import norm
import matplotlib.pyplot as plt
import string
from classSignalData import *

dataFileHistTemplate='MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10'
officialFitHistTemplate='basicBkgFrom4ParamFit'
FIT3_PARS = ['p0','p1','p2']

class dataSet: # each class treats a type of data set
    """each dataSet class  makes predictions using 
        1. GP bkgnd Kernel
        2. GP bkgnd+ signal Kernel
        3. Simple Fit function
        4. Official Fit Function 
        5.SWIFT (Coming soon) """

    def __init__(self, xMinData=None, xMaxData=None, xMinSimpleFit=None, xMaxSimpleFit=None, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate, toy=False, originalSet=None, useScaled=False):
        self.xMinData=xMinData
        self.xMaxData=xMaxData
        self.chi2={}

        if toy==False:
        #Getting Data points
            #print("dataFileName: ",dataFile)
            #dataFileHist=officialFitFile[:-3]
            self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw = getDataPoints(dataFile, dataFileDir, dataFileHist)
            print("print xDataRaw:", self.xRaw)
            self.xRawOffFit, self.yRawOffFit, self.xerrRawOffFit, self.yerrRawOffFit = getDataPoints(officialFitFile, officialFitDir, officialFitHist)
            if useScaled:
                self.yerrRaw=np.sqrt(self.yRaw)
                self.yerrRawOffFit=np.sqrt(self.yRawOffFit)
        #Cutting out the desired range 
            #Data file (For GP Fitting)
            self.xData, self.yData, self.xerrData, self.yerrData = dataCut(xMinData, xMaxData, 0, self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw)  
            if not useScaled:
                self.weight=np.square(self.yerrData)/self.yData
            else:
                self.weight=np.ones(self.yData.shape)
            print("print xData:", self.xData)
            print("yErr:", self.yerrData)
            print("yErrScaled: ", np.sqrt(self.yData))
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
        #self.x_gPPred = np.linspace(np.min(self.xData), np.max(self.xData), 500) ## this is for x_GPpred
        #making x_gpPred that are 2 times the length of x
        self.xPred=[]
        for i in range(len(self.xData)-1):
            self.xPred.append(self.xData[i])
            self.xPred.append(self.xData[i]+(self.xData[i+1]-self.xData[i])/2.)
        #making xerr_gPPred_width 
        self.xPred_width=[]
        self.xPred_width.append(self.xPred[1]-self.xPred[0])
        for i in range(1, len(self.xPred)-1):
            widthHalf1=(self.xPred[i]-self.xPred[i-1])/2.
            widthHalf2=(self.xPred[i+1]-self.xPred[i])/2. 
            width=widthHalf1+widthHalf2
            self.xPred_width.append(width)
        self.xPred_width.append(self.xPred[len(self.xPred)-1]-self.xPred[len(self.xPred)-2])

        self.xPred=np.asarray(self.xPred)
        self.xPred_width=np.asarray(self.xPred_width)

        print("xPred size: ", np.size(self.xPred))  
        print("xPred size: ", np.size(self.xPred_width))  
        print("xPred:", self.xPred)
        print("xPredWidth: ", self.xPred_width)


        #self.xerr_gPPred=self.x_gPPred[1:]-self.x_gPPred[:-1]
        #print("size of xerr_gPPred: ",np.size(self.xerr_gPPred))
        #self.xerr_gPPred=np.append(self.xerr_gPPred, self.xerr_gPPred[-1])
        #print("size of xerr_gPPred: ",np.size(self.xerr_gPPred))


        # boolean counter to see if cetain things are done 
        simpleFitDone=False
        gPBkgKernelFitDone=False
        gPSigPlusBkgKernelFitDone=False
        
######   initialization of points
    #-----Getting Data points
    def initData(self):
        # data points 
        self.xRaw, self.yRaw, self.xerrRaw, self.yerrRaw = getDataPoints(dataFile, dataFileDir, dataFileHist)
        self.xRawOffFit, self.yRawOffFit, self.xerrRawOffFit, self.yerrRawOffFit = getDataPoints(officialFitFile, officialFitDir, officialFitHist)


    #processing: Cutting out the desired range # put this in the Fit part 
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
        #

    def initToy(self):
        pass
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
        print("Check: ", self.x_simpleFit)
        #self.significance_simpleFit, self.chi2['simpleFit']=res_significance(self.y_simpleFit, self.yFit_simpleFit)
        self.significance_simpleFit, self.chi2['simpleFit']=resSigYvonne(self.y_simpleFit, self.yFit_simpleFit)
        if doPrint:
            print("-------------testing Simple Fit------------")
            print("yFit: ", self.yFit_simpleFit)
            print("significance: ", self.significance_simpleFit)

    def gPBkgKernelFit(self, trial=100, useBkgDataResult="", useSimpleFitResult=False,bkgDataHyperParams=None, doPrint=False): 
        """Execute GPBkgKernel Fit"""
        gPBkgKernelFitDone=True 
        self.y_GPBkgKernelFit, self.cov_GPBkgKernelFit, self.bestFit_GPBkgKernelFit=y_bestFitGP(self.xData,self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="bkg", bkgDataParams=None) #bkgDataHyperParam will be None unless set when the function is called 
        self.significance_GPBkgKernelFit, self.chi2['GPBkg']=resSigYvonne(self.yData, self.y_GPBkgKernelFit)
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
        self.y_GPSigPlusBkgKernelFit, self.cov_GPSigPlusBkgKernelFit, self.bestFit_GPSigPlusBkgKernelFit=y_bestFitGP(self.xData,self.xData,self.yData,self.xerrData, self.yerrData,trial, kernelType="sig")
        self.significance_GPSigPlusBkgKernelFit, self.chi2['GPBkgSig']=GPSigPlusBkgKernelFit=resSigYvonne(self.yData, self.y_GPSigPlusBkgKernelFit)
        if doPrint:
            print("-------------testing GP signal plus bkgnd kernel fit ------------")
            print("yFit: ", self.y_GPSigPlusBkgKernelFit)
            print("significance: ", self.significance_GPSigPlusBkgKernelFit)
        
    def yGPSignalReconstructed_dataSBMinusB(self, doPrint=False):
        print("self.bestFit_GPSigPlusBkgKernelFit", self.bestFit_GPSigPlusBkgKernelFit)
        self.MAP_GP, self.MAP_sig, self.MAP_bkg=runGP_SplusB(self.yData, self.xData, self.xerrData,self.xPred, self.xPred_width, self.bestFit_GPSigPlusBkgKernelFit.values())
        if doPrint:
            print("MAP_GP: ", self.MAP_GP)
            print("MAP_sig: ", self.MAP_sig)
            print("MAP_bkg: ", self.MAP_bkg)

    def officialFit(self, doPrint=False):
        """official Fit """
        #need to add a thing to check whether the initial value of the yData and y_officaiFit matches       o 
        self.yFit_officialFit=self.yOffFit
        print("OfficialFit")
        self.significance_officialFit, self.chi2['offFit'] = resSigYvonne(self.yData,self.yOffFit)  
        if doPrint==True:
            print("-----------------Testing Official fit output--------------------")
            print("xFit: ", self.xOffFit)
            print("xData: ", self.xData)
            print("yFit: ", self.yFit_officialFit)
            print("significance", self.significance_officialFit)


    def fitAll(self, trialAll=100,bkgDataParams=None):
        self.bkgData_GPBkgParams=bkgDataParams
        self.simpleFit(trial=trialAll,doPrint=True)
        self.gPBkgKernelFit(trial=trialAll,bkgDataHyperParams=bkgDataParams, doPrint=True)
        self.gPSigPlusBkgKernelFit(trial=trialAll, doPrint=True)
        self.yGPSignalReconstructed_dataSBMinusB(doPrint=True)
        self.officialFit(doPrint=True)



        
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

#making toy

# list of data
#obsolete 
#def makeToyDataSetList(dataSetOriginal, nToy=1):
#    toyDataSetList=[]
#    for i in range(nToy):
#        toyDataSetList.append(dataSet(toy=True, originalSet= dataSetOriginal)) 
#    return toyDataSetList
#
#def listOfChi2(listOfDataSet, fit="simpleFit"):
#    listOfChi2=[]
#    for dataSet in listOfDataSet:
#        listOfChi2.append(dataSet.chi2[fit])
#    return listOfChi2



#def yGPSignalReconstructed_dataSBMinusB(dataSet):
#    MAP_GP, MAP_sig, MAP_bkg=runGP_SplusB(dataSet.yData, data.xData, self.xerrData,self.x_gPPred, self.xerr_gPPred, hyperParams)
#    
        

def fit(config):
    bkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['bkgndDataFile'],dataFileDir=config['bkgFileDir'], dataFileHist=config['bkgDataFileHist'],officialFitFile=config['bkgOffFitFile'])
    bkgndData.fitAll( trialAll=config['trialBkg'])

#--------Dataset: signal plus bkgnd Data
    for mulFac in [1,2, 5, 10, 20]:
        print("multiplication factor", mulFac)

        dataFileString=config['signalPlusBkgFileTemp']

        dataFileString=dataFileString.replace("FFF", str(mulFac))
        print("dataFileString: ",dataFileString)
        
        signalInjectedBkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['sigplusBkgTempDir']+dataFileString, dataFileHist=dataFileString[:-3],officialFitFile=config['sigPlusBkgOffFitFile'])
        signalInjectedBkgndData.fitAll(trialAll=config['trialSigPlusBkg'], bkgDataParams=bkgndData.getGPBkgKernelFitParams())
        #signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
#-----Draw stuff
        title=config['title']+str(mulFac)
        #drawSignalGaussianFit(signalInjectedBkgndData, signalData1, title=title)
        #drawAllSignalFit(signalInjectedBkgndData, signalData1, title=title)
        drawFitDataSet(signalInjectedBkgndData, "SignalinjectedBkgi_"+title)
    drawFitDataSet(bkgndData, config['bkgTitle'], saveTxt=True, saveTxtDir="../txt/BkgData")

    

if __name__=="__main__":        
    print("----------------------")
    print("----------------------")
    print("----------------------")
    print("------test---------")
    print("----------------------")
    print("----------------------")
    print("----------------------")
    
    #configBkg={"bkgndDataFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",
    #        "bkgFileDir": "dijetgamma_g85_2j65",
    #        "bkgDataFileHist": "Zprime_mjj_var",
    #        "bkgOffFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
    #        "xMin": 300,
    #        "xMax": 1500,
    #        "sigplusBkgTempDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/",
    #        "signalPlusBkgFileTemp": "MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulXFFF.h5",
    #        "sigPlusBkgOffFitFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5",
    #        "trialBkg":1,
    #        "trialSigPlusBkg": 1,
    #        "title":"mc_bkgsig_sigmul",
    #        "bkgTitle": "MC_Bkg"}
    #
    #fit(configBkg)

    #configSig={"bkgndDataFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/data/data2016.h5",
    #        "bkgFileDir": "",
    #        "bkgDataFileHist": "Zprime_mjj_var",
    #        "bkgOffFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys//data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
    #        "xMin": 300,
    #        "xMax": 1500,
    #        "sigplusBkgTempDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/data/",
    #        "signalPlusBkgFileTemp": "Data_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulXFFF.h5",
    #        "sigPlusBkgOffFitFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5",
    #        "trialBkg":1,
    #        "trialSigPlusBkg": 1,
    #        "title":"data_bkgsig_sigmul",
    #        "bkgTitle": "Data_Bkg"}

    #fit(configSig)
##################---------------------MC------------------ ####################
#---------DataSet MC

#    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
#    bkgndData.fitAll( trialAll=1)
#
##--------Dataset: signal plus bkgnd Data
#    for mulFac in [1,2, 5, 10, 20]:
#        print("multiplication factor", mulFac)
#
#        dataFileString="MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulXFFF.h5"
#
#        dataFileString=dataFileString.replace("FFF", str(mulFac))
#        print("dataFileString: ",dataFileString)
#        
#        signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/"+dataFileString, dataFileHist=dataFileString[:-3],officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
#        signalInjectedBkgndData.fitAll(trialAll=1, bkgDataParams=bkgndData.getGPBkgKernelFitParams())
#        #signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
##-----Draw stuff
#        title="MC_BkgSig_SigMul"+str(mulFac)
#        #drawSignalGaussianFit(signalInjectedBkgndData, signalData1, title=title)
#        #drawAllSignalFit(signalInjectedBkgndData, signalData1, title=title)
#        drawFitDataSet(signalInjectedBkgndData, "SignalinjectedBkgi_"+title)
#    drawFitDataSet(bkgndData, "MC_BkgDataFit", saveTxt=True, saveTxtDir="txt/BkgData")
#
##################---------------------MC------------------ ####################



#------- Dataset: signal plus bkgnd Data
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    bkgndData.fitAll( trialAll=1)
    #bkgndData=dataSet(300, 1500, 300, 1500, dataFile="data/all/data2016.h5",dataFileDir="", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")
    #bkgndData.fitAll( trialAll=1)

##--------Dataset: signal plus bkgnd Data
    signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5", officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
    signalInjectedBkgndData.fitAll(trialAll=1, bkgDataParams=bkgndData.getGPBkgKernelFitParams())
#
#making a list of toys for the signal plus bkgnd Data 
#-------list of Toy DataSet:bkgnd data
#    signalInjectedBkgndToyList=makeToyDataSetList(signalInjectedBkgndData, 1)
    #bkgDataToyList=makeToyDataSetList(bkgndData, 100)
    #bkgDataToyList.GetChi2List(fit="UA2")
    
    #-----Draw chi2
    #chi2['simpleFit']=listOfChi2(bkgDataToyList, fit="simpleFit")        
    #print("list of chi2: ", chi2['simpleFit'])
#def listOfChi2(listOfDataSet, fit="simpleFit"):
#    bkgData_ToyCollection =toyDataSetCollection(bkgndData, nToys=2) 
#    simpleFitChi2List=bkgData_ToyCollection.GetChi2List("simpleFit")
#    GPBkgFitChi2List=bkgData_ToyCollection.GetChi2List("GPBkg")
#    GPBkgSigFitChi2List=bkgData_ToyCollection.GetChi2List("GPBkgSig")
#
#    print("simpleFitChi2List:",simpleFitChi2List)
#    print("GPBkgFitChi2List: ", GPBkgFitChi2List)
#    print("GPBkgSigFitChi2List:", GPBkgSigFitChi2List)
#


         
##signalData
#making signal data set

    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
    signalData1.print()
##test ToyList
    
#------- drawing stuff
    drawSignalGaussianFit(signalInjectedBkgndData, signalData1)
    drawAllSignalFit(signalInjectedBkgndData, signalData1)
    drawFitDataSet(signalInjectedBkgndData, "TestSignalinjectedBkg")
#    drawFitDataSet(bkgndData, "Bkg", saveTxt=True, saveTxtDir="txt/BkgData")

#
#
### #   def __init__(self, xMinData, xMaxData, xMinSimpleFit, xMaxSimpleFit, dataFile='', dataFileDir='',dataFileHist=dataFileHistTemplate, officialFitFile='', officialFitDir='',officialFitHist=officialFitHistTemplate):
    ##obsolete
    ###ySignalData = y_signalData(signalInjectedBkgndData, bkgndData)
    ##ySignalGPSubtractionFit=y_signalGPSubtractionFit(signalInjectedBkgndData, doPrint=True)
    ##ySignalGaussianFit=y_signalGaussianFit(doPrint=True)
