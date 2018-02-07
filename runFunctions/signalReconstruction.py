from Libplotmeghan import *
from classData import *
from classFitFunction import *
from classSignalData import *

def signalReconstruction(config):
#----Make a bkgnd Data Set
    bkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['bkgDataFile'],dataFileDir=config['bkgDataFileTDir'], dataFileHist=config['bkgDataFileHist'],officialFitFile=config['officialFitFile'])

    bkgndData.fitAll(trialAll=config['trial'])


#----Make a Signal Injected bkgnd Data Set
    signalInjectedBkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['sigBkgDir']+config['sigBkgDataFile'], officialFitFile=config['officialFitFile'])
    signalInjectedBkgndData.fitAll(trialAll=config['trial'], bkgDataParams=bkgndData.getGPBkgKernelFitParams())

#-----Make a signal Data Set 
    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
#TODO add in a signal reconstruction trial #

#----Drawing stuff
    #drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], yFit2=signalData1.sig['bkgOnlyGPPred'], sig=None, title="figure10-1")
    drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'], sig=None, title="figure10")
    drawSignalGaussianFit(signalInjectedBkgndData, signalData1)
    drawAllSignalFit(signalInjectedBkgndData, signalData1)
    drawAllSignalFitYvonne(signalInjectedBkgndData, signalData1,title="gaussian reconstructed")
    drawFitDataSet(signalInjectedBkgndData, "TestSignalinjectedBkg")
    Amp, decay, length, power, sub, p0, p1, p2=bkgndData.bestFit_GPBkgKernelFit.values()
    drawFit(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title="model_gp_withSignal")
    drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title="model_gp_bkg")
    drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], sig=None, title="signal+BkgPred")

if __name__=="__main__":        

    config={"title":"",
            "trial":1,
            "xMin":300,
            "xMax":1500,
            #-------signal +bkg data file
            "bkgDataFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",
            "bkgDataFileTDir":"dijetgamma_g85_2j65",
            "bkgDataFileHist":"Zprime_mjj_var",
            #-------signal +bkg data file
            "sigBkgDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/",
            "sigBkgDataFile":"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5",
    #-------------------------place holder
            "officialFitFile":"data/all/Step1_SearchPhase_Zprime_mjj_var.h5"}

    #mult=["X1", "X2", "X5", "X10", "X20", "X30"]
    #resMass=[250, 350, 400, 450, 500, 550, 750, 950, 1500]
    #ptCut in [50, 100]
    #for coupling in [1,2,3,4]:
    signalReconstruction(config)
