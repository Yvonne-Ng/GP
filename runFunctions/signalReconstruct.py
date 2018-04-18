from Libplotmeghan import *
import json
from classData import *
from classFitFunction import *
from classSignalData import *

from pathlib import Path

from array import *
#from Root import TFile, TH1F
import os.path

#def saveToRoot(fileName,histogramName,x, y, yerr=None):
#    """Saving result bit by bin as a TH1F histogram in a root file"""
#    # x, y nd yerr should be an ARRAY
#    f1=TFile(fileName, "RECREATE") #filename needs to follow a template pattern
#    h1=TH1F(fileName, fileName,x.length(), x)
#    h1.Draw()
#    h1.Save()
#    f1.Save()
#    f1.Close()
#    return 0

#just take in a txt file name instead

def signalReconstruction(config, fixedDataFile=False):
    """fixedDataFile =True if you want to perform all signal rreconstruction as is specified in loop.py on the same data file,
       set it to false if you want to inject signal of each mass point in loop.py and then reconstruct that """

    #---- Make a bkgnd Data Set
    bkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['bkgDataFile'],dataFileDir=config['bkgDataFileTDir'], dataFileHist=config['bkgDataFileHist'],officialFitFile=config['officialFitFile'])
    bkgndData.fitAll(trialAll=config['trial'])

    if not fixedDataFile:
    #----fixedDataFile: Make a Signal Injected bkgnd Data Set
        signalInjectedBkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config["sigBkgDir"]+config['sigBkgDataFile'], officialFitFile=config['officialFitFile'])
        signalInjectedBkgndData.fitAll(trialAll=config['trial'],mass=config['mass'], bkgDataParams=bkgndData.getGPBkgKernelFitParams())
    else:
    #----Non-FixedDataFile:Make a Signal Injected bkgnd Data Set
        print(config['fixedDataDir'])
        signalInjectedBkgndData=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config['fixedDataDir']+config['fixedDataFile'],dataFileDir=config["fixedDataTDir"], dataFileHist=config["fixedDataFileHist"],  officialFitFile=config['officialFitFile'])
        signalInjectedBkgndData.fitAll(trialAll=config['trial'],mass=config['mass'], bkgDataParams=bkgndData.getGPBkgKernelFitParams())
        print("signal+bkgnd Data file: ", config['sigBkgDataFile'])

#-----Make a signal Data Set
    signalData1=signalDataSet(signalInjectedBkgndData, bkgndData,mass=config['mass'], trialAll=config['trial'], configDict=config)

#TODO add in a signal reconstruction trial #

#----Drawing stuff
    #drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], yFit2=signalData1.sig['bkgOnlyGPPred'], sig=None, title="figure10-1")
    #drawFit2
    significance={}
    chi2={}
    significance["sigBkg_GPBkgFit"],chi2["sigBkg_GPBkgFit"] = resSigYvonne(signalInjectedBkgndData.yData, signalData1.sig['bkgOnlyGPPred'], signalData1.sigBkgDataSet.weight)
    significance["sigBkg_GPSigBkgFit"],chi2["sigBkg_GPSigBkgFit"] = resSigYvonne(signalInjectedBkgndData.yData, signalData1.sig['sigPlusBkgPrediction'], signalData1.sigBkgDataSet.weight)
    print("significacne of bkg ", significance["sigBkg_GPBkgFit"])
    #----------draw 2 lines, including the bkgnd kernel prediction and the signal + bkgnd kernel prediction
    drawFit2(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'], sig=[significance["sigBkg_GPBkgFit"],significance["sigBkg_GPSigBkgFit"]],signiLegend=["GP Bkg Significance", "GP bkg+Signal Signifiance"], title=config["title"]+"_figure10")
    #----------draw 3 lines, including the bkgnd reconstructed + the signal reconstructed
    #drawFit3(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'],yFit3=signalData1.sig['bkgOnlyGPPred']+signalData1.sig['custom'], sig=None, title=config["title"]+"_figure10-3Line")
    drawFit3(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.sig['sigPlusBkgPrediction'],yFit3=signalData1.sig['sigPlusBkgPrediction']-signalData1.sig['custom'], sig=None, title=config["title"]+"_figure10-3Line")
    #---------defining significance TODO: move this to the signalDataClass
    significance["sigReconstructedGaussian"], chi2['sigReconstructedGaussian']=resSigYvonne(signalData1.ySigData, signalData1.sig['Gaussian'],signalInjectedBkgndData.weight)
    significance["sigReconstructedCustom"], chi2["sigReconstructedCustom"]=resSigYvonne(signalData1.ySigData, signalData1.sig['custom'],signalInjectedBkgndData.weight)
    sigLegend=["recon. gaussian", "recon. custom"]
    significanceSignal=[significance["sigReconstructedGaussian"], significance["sigReconstructedCustom"]]

#-----------------draw signal only
    #-----------draw signal gaussian fit only
    #drawSignalGaussianFit(signalInjectedBkgndData, signalData1, significanceSignal, sigLegned)

#    drawAllSignalFit(signalInjectedBkgndData, signalData1)
    #---------- draw all signal updated, working version for dijetISR specific needs
    drawAllSignalFitYvonne(signalInjectedBkgndData, signalData1,title=config["title"]+"_gaussian reconstructed",significanceSig=[significance["sigReconstructedGaussian"], significance["sigReconstructedCustom"]],sig=sigLegend)
#---2018-4-10 plotting the figure 10 bkg and the signal bkg

    drawFit3(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=signalData1.sig['bkgOnlyGPPred'], yFit2=signalData1.bkgPred['figure10'],yFit3=signalData1.sigBkgPred['figure10'], legend=["GPBkg+SigKernel Bkg.Pred.", "figure 10 corected bkg", "figure1- corrected bkg+sig"],sig=None, title=config["title"]+"_figure10-3Line_updated")


    #outputFile
    outFileDir="./outputforPython2ToRoot/"
    #outXTitle=outFileDir+"outputX_"+config['title']+".txt"
    #outYTitle=outFileDir+"outputY_"+config['title']+".txt"
    #outYErrTitle=outFileDir+"ouputYerr_"+config['title']+".txt"
    outputTitle=outFileDir+"output_"+config['title']+".txt"

    #arrays for x and y
    #x=array('f', bkgndData.xData)
    #y=array('f', signalData1.sig['bkgOnlyGPPred'])
#list for x and y
    x=signalInjectedBkgndData.xData
    y=signalData1.sig['bkgOnlyGPPred']
    #TODO error for y require pseudo experiments
#---- json output dump
    with open(outputTitle, 'w') as outFile:
        xround=[float(int(xEle)) for xEle in x]
        print("xround: ", xround)
        json.dump({"config":config['configFile'],"x":xround, "y":y.tolist()}, outFile)

#----saving to txt bs
    #outFilex=open(outXTitle, "w")
    #for i in x :
    #    outFilex.write(str(i)+"   ")

    #outFilex.close()

    #outFiley=open(outYTitle, "w")
    #for i in y :
    #    outFiley.write(str(i)+"   ")
    #outFiley.close()

    #outConfig=open(outputConfig, "w")



#-----------------draw data points only
    #---------- draw signal +bkgnd MC
    #drawFitDataSet(signalInjectedBkgndData, config["title"]+"_TestSignalinjectedBkg")
    Amp, decay, length, power, sub, p0, p1, p2=bkgndData.bestFit_GPBkgKernelFit.values()
    #drawFit(xData=signalInjectedBkgndData.xData,yerr=signalInjectedBkgndData.yerrData, yData=signalInjectedBkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title=config["title"]+"_model_gp_withSignal")
    #drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=model_gp((p0,p1,p2), bkgndData.xData,bkgndData.xerrData), sig=None, title=config["title"]+"_model_gp_bkg")
    #drawFit(xData=bkgndData.xData,yerr=bkgndData.yerrData, yData=bkgndData.yData, yFit=signalData1.sig['sigPlusBkgPrediction'], sig=None, title=config["title"]+"_signal+BkgPred")
if __name__=="__main__":
    config={"title":"testY",
            "trial":30,
            "xMin":300,
            "xMax":1500,
            #-------signal +bkg data file
            "bkgDataFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",
            "bkgDataFileTDir":"dijetgamma_g85_2j65",
            "bkgDataFileHist":"Zprime_mjj_var",
            #-------signal +bkg data file
            "mass": 500, #GeV
            "sigBkgDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/",
            "sigBkgDataFile":"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5",
            #------signal template
            "sigTemplateFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/signal/reweighted_Signal_dijetgamma_g85_2j65_36p",
            "sigTemplateHist": "reweighted_Ph100_ZPrimemR500_gSM0p3",

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
    titleTemplate="testY"
    sigHistTemplate= "reweighted_Ph100_ZPrimemR500_gSM0p3"
    for ptCut in [50, 100]:
        for mult in [1,5,10, 20]:
            for resMass in [350, 400, 450, 500, 550, 750, 950]:
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
                        config['title']=titleTemplate+"_Ph"+str(ptCut)+"_mR"+str(resMass)+"_gSM"+str(coupling)+"_Mul"+str(mult)
                        config['sigBkgDataFile']=newTemplate
                        signalReconstruction(config)
                    else:
                        print("file does not exist!")

        #change the amplitude in signal kernel
    #    for resMass in [250, 350, 400, 450, 500, 550, 750, 950, 1500]:
    #        for ptCut in [50,100]:
                #Changes the Fit
                #only print out the one that got picked up
