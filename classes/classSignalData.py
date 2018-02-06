#!/usr/bin/env python3
#using the bkgnd data of MC reweighted of dijetgamma_g85_2j65
from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp, ExpSquaredCenteredKernel#, ExpSquaredKernel
from iminuit import Minuit
import scipy.special as ssp
import inspect
import random

#importing functions from Libplotmeghan
from classDataSetCollection import *
from Libplotmeghan import get_xy_pts,getDataPoints,dataCut, removeZeros,makeToys,y_bestFit3Params, y_bestFitGP, res_significance, significance, runGP_SplusB,logLike_3ffOff, gaussian, resSigYvonne,logLike_gp_sigRecon, fit_gp_sigRecon, model_3param,logLike_gp_sig_fixedH, fit_gp_sig_fixedH_minuit,logLike_gp_sig_fixedH, sig_model,logLike_gp_customSigTemplate, fit_gp_customSig_fixedH_minuit,customSignalModel



#import function from drawStuff
from drawStuff import drawSignalGaussianFit, drawSignalSubtractionFit, drawFitDataSet, drawAllSignalFit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#for the gaussian fit 
from scipy.stats import norm
import matplotlib.pyplot as plt
import string




class signalDataSet():
    def __init__(self,signalBkgDataSet, bkgDataSet , useScaled=False): 
        #make sure the size of both data matches 
        # finding the raw data points
        if np.any(np.not_equal(signalBkgDataSet.xData,bkgDataSet.xData)):
            print("Error, xBkg and xBkgbk value are different")
            return  1
        else :
            self.sigBkgDataSet=signalBkgDataSet
            self.ySigData = signalBkgDataSet.yData-bkgDataSet.yData
            self.xSigData = signalBkgDataSet.xData
            self.bkgDataSet=bkgDataSet
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

        self.sig={}
        # SIGNAL RECONSTRUCTION USING OFFICIAL CODE 
        self.fixedBkgKernelHyperParams=bkgDataSet.bestFit_GPBkgKernelFit
        print("bkggdKernelHyperParams:",bkgDataSet.bestFit_GPBkgKernelFit)
        self.sig['sigPlusBkgPrediction'], self.sig['Sig_GPSigKernel'], self.sig['bkgOnlyGPPred']=self.doReconstructedSignal("GPSigKernel")
        self.sig['Gaussian']=self.doReconstructedSignal("Gaussian")
        self.sig['custom']=self.doReconstructedSignal("custom")

        #print best_vals

    def doReconstructedSignal(self,option="GPSigKernel", trial=1):
        self.trial=trial
        if option=="GPSigKernel":
            return self.reconstructSignalGPSignalKernel(self.fixedBkgKernelHyperParams,trial)
        if option=="Gaussian":
            return self.reconstructSignalGaussianTemplate(self.fixedBkgKernelHyperParams)
        if option=="custom":
            return self.reconstructSignalCustomSignalTemplate(self.fixedBkgKernelHyperParams)
        

    def reconstructSignalGPSignalKernel(self, fixedBkgKernelHyperParams,trial):
        Amp, decay, length, power, sub, p0, p1, p2 = fixedBkgKernelHyperParams.values()
        lnProb = logLike_gp_sigRecon(self.sigBkgDataSet.xData,self.sigBkgDataSet.yData, self.sigBkgDataSet.xerrData, fixedBkgKernelHyperParams,weight= self.sigBkgDataSet.weight)
        print("test: ",lnProb(10000000, 500, 100))
        #bestval, best_fit_new = fit_gp_fitgpsig_minuit( trial, lnProb, list(self.bkgDataSet.getGPBkgKernelFitParams().values()), False)
        bestval, best_fit_new = fit_gp_sigRecon( lnProb, trial=1, Print=False)
        A, mass, tau = best_fit_new
        kernel2 = A * ExpSquaredCenteredKernel(m = mass, t = tau)
        kernel1 = Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d=sub)
        kernel = kernel1 + kernel2
        gp = george.GP(kernel)
        gp.compute(self.sigBkgDataSet.xData, self.sigBkgDataSet.yerrData)
        meanGPp = gp.predict( self.sigBkgDataSet.yData - model_3param((p0,p1,p2),self.sigBkgDataSet.xData,self.sigBkgDataSet.xerrData), self.sigBkgDataSet.xData)[0]
        meanGP = meanGPp + model_3param((p0,p1,p2),self.sigBkgDataSet.xData,self.sigBkgDataSet.xerrData)
        gp2 = george.GP(kernel)
        gp2.compute(self.sigBkgDataSet.xData, np.sqrt(self.sigBkgDataSet.yData))
        K1 = kernel1.get_value(np.atleast_2d(self.sigBkgDataSet.xData).T)
        K2 = kernel2.get_value(np.atleast_2d(self.sigBkgDataSet.xData).T)
        #print("model_3param", self.sigBkgDataSet.yData-model_3param((p0,p1, p2), self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData) )  
        #print("self.sigBkgDataSet.yData", self.sigBkgDataSet.yData)
        #print("K1: ", K1)
        #print("gpSolver:", gp2.solver.apply_inverse(self.sigBkgDataSet.yData- model_3param((p0,p1,p2),self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData)))


        #print("np.dot", np.dot(K1, gp2.solver.apply_inverse(self.sigBkgDataSet.yData- model_3param((p0,p1,p2),self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData))))
        mu1 = np.dot(K1, gp2.solver.apply_inverse(self.sigBkgDataSet.yData- model_3param((p0,p1,p2),self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData))) + model_3param((p0,p1, p2), self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData)
        mu2 = np.dot(K2, gp2.solver.apply_inverse(self.sigBkgDataSet.yData- model_3param((p0,p1,p2),self.sigBkgDataSet.xData, self.sigBkgDataSet.xerrData)))
        return mu1, mu2, meanGP
        
    def reconstructSignalGaussianTemplate(self, fixedBkgKernelHyperParams):
         lnProb = logLike_gp_sig_fixedH(self.sigBkgDataSet.xData,self.sigBkgDataSet.yData, self.sigBkgDataSet.xerrData,self.fixedBkgKernelHyperParams)
         bestval, best_fit = fit_gp_sig_fixedH_minuit(lnProb, False)
         print("got here")
         #if np.isinf(bestval): continue 
         N, M, W = best_fit
         ySig=sig_model(self.sigBkgDataSet.xData, N, M, W, self.sigBkgDataSet.xerrData)
         return ySig

    def reconstructSignalCustomSignalTemplate(self,fixedBkgKernelHyperParams):
        #f1=TFile.Open("/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/signal/reweighted_Signal_dijetgamma_g85_2j65_36p1fb.root", "READ")
        #h1=f1.Get("reweighted_Ph100_ZPrimemR500_gSM0p3")
        #h1=h1.Scale(1/h1.Integral())
        #print("h1 area: ", h1.Integral())
        #self.signalTemplate=h1

        xTemplate, yTemplate, xerrTemp, yerrTemp = getDataPoints("/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/signal/reweighted_Signal_dijetgamma_g85_2j65_36p1fb.h5", "", "reweighted_Ph100_ZPrimemR500_gSM0p3")
        xTemplate, yTemplate, xerrTemp, yerrTemp =dataCut(300, 1500, 0, xTemplate, yTemplate, xerrTemp, yerrTemp )
        norm=np.ndarray.sum(yTemplate)
        print("xData", self.sigBkgDataSet.xData)
        print("xTempl: ", xTemplate)
        print("yTemplate: ", yTemplate)
        print("norm: ", norm)

        yNorm=yTemplate/norm
        print("yNom: ", yNorm)
        lnProb=logLike_gp_customSigTemplate(self.sigBkgDataSet.xData,self.sigBkgDataSet.yData, self.sigBkgDataSet.xerrData,xTemplate, yNorm,self.fixedBkgKernelHyperParams)
        bestval, best_fit=fit_gp_customSig_fixedH_minuit(lnProb)
        N=best_fit
        print("got here")
        ySig=customSignalModel(N, yNorm)
        print("got there")
        return ySig
        
    def print(self):
        print("xSigData: ",self.xSigData)
        print("ySigData: ",self.ySigData)
        print ("GP subtraction fit y : ", self.yGPSubtractionFit )
        print ("GP siginificaance : ", self.gpSubtractionSignificance)
        print("Gaussian Fit : ", self.yGaussianFit)
        print ("Gaussian significance: ", self.gaussianFitSignificance)


