
from Libplotmeghan import *
import json
from classData import *
from classFitFunction import *
from classSignalData import *

from pathlib import Path

from array import *
#from Root import TFile, TH1F
import os.path
from random import *
from numpy.random import poisson
from drawStuff import *

# code written to compare the bkg kernel estimation between signal injected and signal not injected#

#Grab different kinds of injected signal file

def findInjectionMultiplication(fileName):
    pos0=fileName.find("mulX")+len("mulX")
    pos1=fileName[pos0:].find(".")+pos0
    print("pos0", pos0)
    print("pos1", pos1)
    return fileName[pos0:pos1]

def doPseudoExperiment(yFitList, weight):
    #seed = randint(1, 100)
    #binSeed =  int( round(seed*1e5))
    #rand3 = ROOT.TRandom3(binSeed)
    yFitFluc=[]
    for i in range(len(yFitList)):
        yFitUnweighted=yFitList[i]/weight[i]
        yFitFluc.append(int(round(poisson(yFitUnweighted))))
    return yFitFluc



def chi2ndfCalculation(dataYList, fitYList, weightList=None):
    chi2=0
    if weight is not None:
        for i in range(len(fitYList)):
            print("dataYList[i]: ",dataYList[i])
            print("weightList[i]: ",weightList[i])
            print("fitYList[i]: ", fitYList[i])
            chi2=chi2+np.square((dataYList[i]/weightList[i])-(fitYList[i]/weightList[i]))/(dataYList[i]/weightList[i])
    else:
        for i in range(len(fitYList)):
            chi2=chi2+np.square(dataYList[i]-fitYList[i])/dataYList[i]
    return chi2/(len(dataYList)-1)


if __name__=="__main__":
    injectedSigFileDir="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC"
    injectedSignalFiles=["MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5"   ,"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX2.h5"   ,"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX30.h5",
        "MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5"  ,"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX20.h5",  "MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX5.h5"]

    bkgFileDir="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC"
    bkgFile="MC_Dan.h5"

    #process Data points
    #----initialization
    x={}
    y={}
    xerr={}
    yerr={}
    weight={}
    mean={}
    cov={}
    bestFit={}
    significance={}
    chi2={}
    chi2ndf={}
    tagSet=[]
    #----process bkg data
    x["bkg"], y["bkg"], xerr["bkg"], yerr["bkg"]=getDataPoints(bkgFileDir+"/"+bkgFile,"dijetgamma_g85_2j65","Zprime_mjj_var")
    x["bkg"], y["bkg"], xerr["bkg"],yerr["bkg"]=dataCut(300, 1500, 0, x["bkg"], y["bkg"], xerr["bkg"],yerr["bkg"])
    #calculate chi2

    #----Fitting bkgData

    #self.weight=np.square(self.yerrData)/self.yData
    weight["bkg"]=np.square(yerr["bkg"])/y["bkg"]
    mean["bkg"], cov["bkg"], bestFit["bkg"]= y_bestFitGP(x["bkg"], x["bkg"], y["bkg"], xerr["bkg"] , yerr["bkg"], 15, kernelType="bkg")
    significance["bkg"],chi2["bkg"] = resSigYvonne(y["bkg"], mean["bkg"], weight["bkg"])
    print("chi2/ndf is for bkg is : ", chi2ndfCalculation(y["bkg"], mean["bkg"], weight["bkg"]))
    tagSet.append("bkg")


    #----process signal data
    for injectedFile in injectedSignalFiles:
        mult=findInjectionMultiplication(injectedFile)
        tag="sigInj_"+mult
        tagSet.append(tag)
        x[tag], y[tag], xerr[tag], yerr[tag]=getDataPoints(injectedSigFileDir+"/"+injectedFile,"",injectedFile[:-3])
        x[tag], y[tag], xerr[tag],yerr[tag]=dataCut(300, 1500, 0, x[tag], y[tag], xerr[tag],yerr[tag])
        weight[tag]=np.square(yerr[tag])/y[tag]

        #----fitting with bkg kernel for injected signal
        print("mass: ", 500, " mult: X", mult)
        mean[tag], cov[tag], bestFit[tag]= y_bestFitGP(x[tag], x[tag], y[tag], xerr[tag] , yerr[tag], 2, kernelType="bkg")
        #---drawing the comaprison with bkg fit
#def drawFit2(xData=None, yerr=None, yData=None, yFit=None,yFit2=None, sig=None, signiLegend=None, title=None, saveTxt=False, saveTxtDir=None):
        #significance["sigBkg_GPBkgFit"],chi2["sigBkg_GPBkgFit"] = resSigYvonne(signalInjectedBkgndData.yData, signalData1.sig['bkgOnlyGPPred'], signalData1.sigBkgDataSet.weight)
        significance[tag],chi2[tag] = resSigYvonne(y[tag], mean[tag], weight[tag])

        print("chi2/ndf is for sigInject X",tag,": ", chi2ndfCalculation(y[tag], mean[tag], weight[tag]))
        drawFit2(xData=x[tag], yerr=yerr[tag], yData=y[tag], yFit=mean["bkg"],yFit2=mean[tag], sig=[significance["bkg"], significance[tag]], signiLegend=["bkgkernel Bkg data", "bkg kernel injected data"], title="Fit2_"+tag, saveTxt=False, saveTxtDir=None)

#------make a set of pseudo experiment for all the fits
    for tag in tagSet:
        originalTag=tag
        pseudoTag=tag+"Pseudo"

        y[pseudoTag]={}
        weight[pseudoTag]={}
        mean[pseudoTag]={}
        cov[pseudoTag]={}
        bestFit[pseudoTag]={}
        significance[pseudoTag]={}
        chi2[pseudoTag]={}
        chi2ndf[pseudoTag]=[]

        for i in range(25):
            #---fluctuate the the y value of the previous fit result
            #--- and perform the fit again
            y[pseudoTag][i]=[]
            weight[pseudoTag][i]=[]
            mean[pseudoTag][i]=[]
            cov[pseudoTag][i]=[]
            bestFit[pseudoTag][i]=[]
            significance[pseudoTag][i]=[]
            chi2[pseudoTag][i]=[]
            #chi2ndf[pseudoTag][i]=[]

            y[pseudoTag][i]=doPseudoExperiment(mean[originalTag],weight[originalTag])
            weight[pseudoTag][i]=(np.square(yerr[originalTag])/y[pseudoTag][i]).tolist()
            #print("weight[pseudoTag][i]: ", weight[pseudoTag][i])
            #print("weight type: ", type(weight[pseudoTag][i]))
            mean[pseudoTag][i], cov[pseudoTag][i], bestFit[pseudoTag][i]= y_bestFitGP(x[tag], x[tag], y[pseudoTag][i], xerr[tag] , yerr[tag], 2, kernelType="bkg")
            significance[pseudoTag][i],chi2[pseudoTag][i] = resSigYvonne(y[pseudoTag][i], mean[pseudoTag][i], weight[pseudoTag][i])
            #print("length of weight: ", len(weight[pseudoTag][i]))
            #print("length of y: ", len(y[tag][i]))

            if np.isnan(chi2ndfCalculation(y[pseudoTag][i], mean[pseudoTag][i], weight[pseudoTag][i])):
                chi2ndf[pseudoTag].append(20)
            else:
                chi2ndf[pseudoTag].append(chi2ndfCalculation(y[pseudoTag][i], mean[pseudoTag][i], weight[pseudoTag][i]))

        #print( "chi2/ndf for tag: ", tag, chi2ndf[pseudoTag])
        print( "mean chi2/ndf for tag: ", tag, np.mean(chi2ndf[pseudoTag]))
    drawChi2Dist("chi2/ndf: ", chi2ndf, tagSet)
