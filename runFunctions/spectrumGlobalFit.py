#import numpy as np
#import george
#from iminuit import Minuit
#import scipy.special as ssp
#import inspect
##from lib-plotmeghan import *
#from Libplotmeghan import *
#from classData import *
#from drawStuff import drawFit
from classFitFunction import *

fitFunctionFromCode=["UA2", "std4Params", "param3fit", "param5fit"]
fitFunctionCodeStd={"UA2": 1, "std4Params": 4, "param3fit": 3, "param5fit": 5}
nParam=[4, 4, 3,5]

def spectrumGlobalFit(config):
    print(config)
    mjjData=dataSet(config['xMinFit'], config['xMaxFit'], config['xMinGP'], config['xMaxGP'], dataFile=config['dataFile'],dataFileDir=config['dataFileTDir'], dataFileHist=config['dataFileHist'],officialFitFile=config['officialFitFile'], useScaled=config['useScaled'])
    spectrumFitLog=["----Log for %s----"%(config["title"])]
    # Create a fit function
#----find weight hist
    if not config['useScaled']:
        config['weightHist']=mjjData.weight
    else:
        config['weightHist']=None

    Fit=FitFunction(config['fitFunction'])
    Fit.grabNProcessData(mjjData.xMinData, mjjData.xMaxData, mjjData.xData, mjjData.yData, mjjData.xerrData, mjjData.yerrData, useScaled=config['useScaled'], weight=config['weightHist'])
    print("weight: ",mjjData.weight)

    yFit=Fit.doFit(initFitParam=config['initFitParam'], initRange=config['initRange'], trial=100, useScaled=config['useScaled'])
    #-- saving the log---
    #spectrumFitLog.append("fit function: %s"%fitFunctionFromCode[config["fitFunction"]])
    #spectrumFitLog.append("fit range: %d - %d"%(config["xMinFit"], config["xMaxFit"]))
    spectrumFitLog.append("minimum -LL: %d"%Fit.minimumLLH)
    if Fit.minimumLLH>900:
        spectrumFitLog.append("possible Bad Fit")
    for i in range(len(Fit.bestFitParams)):
        spectrumFitLog.append("parameter{0}   {1}".format(i+1, Fit.bestFitParams[i]))
    spectrumFitLog.append("--------------------------------------\n")

    print("yFit: ",yFit)
    print("yData: ", mjjData.yData)

    sig, chi2=resSigYvonne(yFit,mjjData.yData, config['weightHist'])
    drawFit(mjjData.xData, mjjData.yerrData, mjjData.yData, yFit, sig,config['title'])
    return spectrumFitLog

if __name__=="__main__":
#-----------a template config file -------#
    config={#-----Title
            "title": "trijet1btagged-UA2",
            "useScaled": False,
            #-----fit range
            #"xMinFit": 330,
            #"xMaxFit": 1359,
            #"xMinGP": 330,
            #"xMaxGP": 1359,
            "xMinFit": 300,
            "xMaxFit": 1500,
            "xMinGP": 300,
            "xMaxGP": 1500,
            #-----Spectrum file input
            "dataFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/reweighted_hist-background_ABCD_trijet.h5",
            "dataFileTDir": "",
            "dataFileHist": "background_mjj_var",
            #------put some placeholder file here
            "officialFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
            #-----Fit function
            "fitFunction": 0, #0: UA2; 1: 4 params
            "initFitParam": None, #None(default): (9.6, -1.67, 56.87,-75.877 )
            "initRange": None} #None(default): [(-100000, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)]
    spectrumGlobalFit(config)


