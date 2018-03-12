import array
from ROOT import TFile,TH1F
import json
import importlib
import os

def saveTxtToRoot(inDir, inFile, rootName, histName):
    """saving txt x and y that are outputs from runFunction.signalReconstruction to a root file in a histogram. The histogram will be an input for the setLimitOneMassPoint"""
   #-----grabing with json
    with open(inDir+inFile) as jsonInput:
        inputData = json.load(jsonInput)
        print(inputData)
        print("config file: ", inputData["config"])

    loopConfigFile=importlib.import_module(inputData["config"])
    print(loopConfigFile.config)

#--adding end bound for x
    x=inputData["x"]
    diff=x[-1]-x[-2]
    x.append(x[-1]+diff)
    x=array.array("f",x)

    y=inputData["y"]
    y=array.array("f",y)
    #---- Size check
    if len(x)-1 !=len(y):
        print "x and y file have array of different sizes!"
        return -1

    rootFile=TFile(rootName+".root", "RECREATE")
    bkgFitHist=TH1F(histName, histName, len(x)-1, x)
    for i in range(len(y)):
        bkgFitHist.SetBinContent(i+1, y[i])
        #print(i)
        #TODO set error from pseudo experiemnt
    bkgFitHist.Write()
    rootFile.Close()
def getTitleFromFileName(fileName):
    """Getting the "title of the run by figuring it out through the file name"""
    pass
def makeJsonNameFromTemplate(template, ptCut, resMass, coupling, mult):
    pos0={}
    pos1={}
    pos0["Ph"]=template.find("Ph")+2
    pos1["Ph"]=template[pos0["Ph"]:].find("_")+pos0["Ph"]
    pos0["mR"]=template.find("mR")+2
    pos1["mR"]=template[pos0["mR"]:].find("_")+pos0["mR"]
    pos0["gSM"]=template.find("gSM")+3
    pos1["gSM"]=template[pos0["gSM"]:].find("_")+pos0["gSM"]
    pos0["Mul"]=template.find("Mul")+3
    pos1["Mul"]=template[pos0["Mul"]:].find(".")+pos0["Mul"]
    print("mult: ", mult)
    newFileName=template[:pos0["Ph"]]+str(ptCut)+template[pos1["Ph"]:pos0["mR"]]+str(resMass)+template[pos1["mR"]:pos0["gSM"]]+str(coupling)+template[pos1["gSM"]:pos0["Mul"]]+str(mult)+template[pos1["Mul"]:]
    print(newFileName)
    return newFileName




#output_testprintTXT_Ph100_mR750_gSM2_Mul10.txt




def loopSaveTxtRoot(loopConfig, inputDir, inputTxtTemplate):
    """looping through the loop.py info for the file"""
    #load loop config
    loopConfigFile=importlib.import_module(loopConfig)
    # find title root name
    title=findTitleFromTemplate(inputTxtTemplate)
    #loop through the details
    for ptCut in loopConfigFile.config['ptCuts']:
        for mult in loopConfigFile.config['mults']:
            for resMass in loopConfigFile.config['resMasses']:
                for coupling in loopConfigFile.config['couplings']:
                    #------create the file name
                    txtFileName=makeJsonNameFromTemplate(inputTxtTemplate, ptCut, resMass, coupling,mult)
                    #------Check if the file exist in the specified directory
                    print("inputDir: ", inputDir)
                    print("txtFileName: ", txtFileName)
                    if os.path.isfile(inputDir+txtFileName):
                        #-------print info about the file
                        print("fileName ", txtFileName)
                        print("fileExist!")
                        #------Set title of the output file
                        saveTxtToRoot(inputDir, txtFileName, title, "bkgEstFromGP")
                    else:
                        print("fileName ", txtFileName)
                        print("file does not exist!")

def findTitleFromTemplate(template):# for root file name
    pos0=template.find("_")
    pos1=template.find(".")+pos0
    title=template[pos0+1: pos1]
    print("title: ", title)
    return title

if __name__=="__main__":
#------Tesing code for saveTxtToRoot function defined in this file-----
    #rootName=
#----initialization
    template={"inputTxtDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/outputforPython2ToRoot/",
            "inputFile":"output_testprintTXT_Ph100_mR750_gSM2_Mul10.txt"}

    with open(template["inputTxtDir"]+template["inputFile"]) as jsonInput:
        inputData = json.load(jsonInput)
        print(inputData)
        print("config file: ", inputData["config"])

#----config for loop made, run loop
    loopSaveTxtRoot(inputData["config"], template["inputTxtDir"], template["inputFile"])
  #  saveTxtToRoot(config['inputTxtDir'],config['inputFile'], "something.root", "hist")
