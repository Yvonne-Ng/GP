from Libplotmeghan import *
from classData import *
from classFitFunction import *
from classSignalData import *

def signalReconstruction():
#----Make a bkgnd Data Set
    bkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",dataFileDir="dijetgamma_g85_2j65", dataFileHist="Zprime_mjj_var",officialFitFile="data/all/Step1_SearchPhase_Zprime_mjj_var.h5")

    bkgndData.fitAll(trialAll=20)


#----Make a Signal Injected bkgnd Data Set
    signalInjectedBkgndData=dataSet(300, 1500, 300, 1500, dataFile="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5", officialFitFile="data/all/Step1_SearchPhase_MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX1.h5")
    signalInjectedBkgndData.fitAll(trialAll=20, bkgDataParams=bkgndData.getGPBkgKernelFitParams())

#-----Make a signal Data Set 
    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData)
    print("signalData1.sig['Gaussian']", signalData1.sig['Gaussian'])
    

    #--do someth kind of fit 

    #signalData1.print()

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
    signalReconstruction()
