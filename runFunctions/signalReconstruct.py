from Libplotmeghan import *
from classData import *
from classFitFunction import *
from classSignalData import *

from pathlib import Path

import os.path

def signalReconstruction(config):
#----Make a bkgnd Data Set
    bkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['bkgDataFile'],dataFileDir=config['bkgDataFileTDir'], dataFileHist=config['bkgDataFileHist'],officialFitFile=config['officialFitFile'])

    bkgndData.fitAll(trialAll=config['trial'])


#----Make a Signal Injected bkgnd Data Set
    signalInjectedBkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['sigBkgDir']+config['sigBkgDataFile'], officialFitFile=config['officialFitFile'])
    signalInjectedBkgndData.fitAll(trialAll=1, bkgDataParams=bkgndData.getGPBkgKernelFitParams())

#-----Make a signal Data Set 
    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData,trialAll=1)
#TODO add in a signal reconstruction trial #

#----Drawing stuff
    #drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], yFit2=signalData1.sig['bkgOnlyGPPred'], sig=None, title="figure10-1")
    drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'], sig=None, title=config["title"]+"_figure10")
    drawFit3(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'],yFit3=signalData1.sig['bkgOnlyGPPred']+signalData1.sig['custom'], sig=None, title=config["title"]+"_figure10-3Line")
    drawSignalGaussianFit(signalInjectedBkgndData, signalData1)
    drawAllSignalFit(signalInjectedBkgndData, signalData1)
    drawAllSignalFitYvonne(signalInjectedBkgndData, signalData1,title=config["title"]+"_gaussian reconstructed")
    drawFitDataSet(signalInjectedBkgndData, config["title"]+"_TestSignalinjectedBkg")
    Amp, decay, length, power, sub, p0, p1, p2=bkgndData.bestFit_GPBkgKernelFit.values()
    drawFit(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title=config["title"]+"_model_gp_withSignal")
    drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title=config["title"]+"_model_gp_bkg")
    drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], sig=None, title=config["title"]+"_signal+BkgPred")

if __name__=="__main__":        
    config={"title":"testY",
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
            #-------output
            "outputDir":"./results",
    #-------------------------place holder
            "officialFitFile":"data/all/Step1_SearchPhase_Zprime_mjj_var.h5"}

   # mult=["X1", "X2", "X5", "X10", "X20", "X30"]
   # resMass=[250, 350, 400, 450, 500, 550, 750, 950, 1500]
   # ptCut in [50, 100]
   # for coupling in [1,2,3,4]:

#----------------------Loop stuff------------------------"
config['sigBkgDir']="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/feb2018/"
template="MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5"
for ptCut in [50, 100]:
    for mult in [1,10, 20]:
        for resMass in [250, 350, 400, 450, 500, 550, 750, 950, 1500]:
            for coupling in [1,2,3,4]:
                posCut=template.find("Ph")+2
                posMass=template.find("mR")+2
                posCoupling=template.find("gSM0p")+5
                posMul=template.find("mulX")+4
                pos_1=template[posCut:].find("_")+ posCut
                pos_2=template[posMass:].find("_")+posMass
                pos_3=template[posCoupling:].find("_")+posCoupling
                newTemplate=template[:posCut]+str(ptCut)+template[pos_1:posMass]+str(resMass)+template[pos_2:posCoupling]+str(coupling)+template[pos_3:posMul]+str(mult)+".h5"
                print("newTemplate: ", newTemplate)
                if os.path.isfile(config['sigBkgDir']+newTemplate):
                    print("fileExist!")
                    config['title']=config['title']+"_Ph"+str(ptCut)+"_mR"+str(resMass)+"_gSM"+str(coupling)+"_Mul"+str(mult)
                    config['sigBkgDataFile']=newTemplate
                    signalReconstruction(config)
                else:
                    print("file does not exist!")

    #change the amplitude in signal kernel
#    for resMass in [250, 350, 400, 450, 500, 550, 750, 950, 1500]:
#        for ptCut in [50,100]:
            #Changes the Fit 
            #only print out the one that got picked up
