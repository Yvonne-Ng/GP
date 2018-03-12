#import signalReconstruction
from runFunctions.signalReconstruct import signalReconstruction
from signalReconstructionConfig import *
import argparse
import importlib
from pathlib import Path
import os.path

def inputFileNameCreate(template, ptCut, mult, resMass, coupling):
    posCut=template.find("Ph")+2
    posMass=template.find("mR")+2
    posCoupling=template.find("gSM0p")+5
    posMul=template.find("mulX")+4
    pos_1=template[posCut:].find("_")+ posCut
    pos_2=template[posMass:].find("_")+posMass
    pos_3=template[posCoupling:].find("_")+posCoupling
    newTemplate=template[:posCut]+str(ptCut)+template[pos_1:posMass]+str(resMass)+template[pos_2:posCoupling]+str(coupling)+template[pos_3:posMul]+str(mult)+".h5"
    print("newTemplate: ", newTemplate)
    return(newTemplate)

def signalHistNameCreate(template, ptCut, resMass, coupling):
    posCut=template.find("Ph")+2
    posMass=template.find("mR")+2
    posCoupling=template.find("gSM0p")+5
    pos_1=template[posCut:].find("_")+ posCut
    pos_2=template[posMass:].find("_")+posMass
    pos_3=template[posCoupling:].find("_")+posCoupling
    newTemplate=template[:posCut]+str(ptCut)+template[pos_1:posMass]+str(resMass)+template[pos_2:posCoupling]+str(coupling)
    print(newTemplate)
    return newTemplate

if __name__=="__main__":
    #-----Argument Parser
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", default="signalReconstructionConfig.template")
    parser.add_argument("--loopConfig", default="signalReconstructionConfig.loop")
    args=parser.parse_args()
    print(args.config)
    #-----converting string to .py
        #-------initConfig file is the initial file for single loop
    initConfigFile=importlib.import_module(args.config)
    loopConfigFile=importlib.import_module(args.loopConfig)
    #-----Printing the dictionary as a check
    print(initConfigFile.config)
    print(loopConfigFile.config)
    initConfigFile.config["configFile"]=args.loopConfig


#-----------loop specific template option TODO: throw this into loop templates
    initConfigFile.config['sigBkgDir']="/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/feb2018/"
    template=initConfigFile.config['sigBkgDataFile']
                #-----"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5"
    #-----Holder for the template of the title for each input
    titleTemplate=initConfigFile.config['title']
    sigTemplateHist=initConfigFile.config['sigTemplateHist']
    for ptCut in loopConfigFile.config['ptCuts']:
        for mult in loopConfigFile.config['mults']:
            for resMass in loopConfigFile.config['resMasses']:
                for coupling in loopConfigFile.config['couplings']:
                    #------create the new input .h5 file name
                    newTemplate=inputFileNameCreate(template, ptCut, mult, resMass, coupling)
                    signalHist=signalHistNameCreate(sigTemplateHist, ptCut, resMass, coupling)
                    initConfigFile.config['mass']=resMass
                    #------Check if the file exist in the specified directory
                    if os.path.isfile(initConfigFile.config['sigBkgDir']+newTemplate):
                        #-------print info about the file
                        print("fileName ", newTemplate)
                        print("fileExist!")
                        #------Set title of the output file
                        initConfigFile.config['title']=titleTemplate+"_Ph"+str(ptCut)+"_mR"+str(resMass)+"_gSM"+str(coupling)+"_Mul"+str(mult)
                        initConfigFile.config['sigTemplateHist']=signalHist
                        #-----Set the input file name
                        initConfigFile.config['sigBkgDataFile']=newTemplate
                        #----Run signal reconstruction
                        signalReconstruction(initConfigFile.config)
                    else:
                        print("fileName ", newTemplate)
                        print("file does not exist!")

