import array
from ROOT import TFile,TH1F
import json
import importlib
import os

def saveTxtToRoot(inDir, inFile, outDir, rootName, histName):
    """saving txt x and y that are outputs from runFunction.signalReconstruction to a root file in a histogram. The histogram will be an input for the setLimitOneMassPoint"""
   #-----grabing with json
    print "file: ", inDir+inFile
    with open(inDir+inFile) as jsonInput:
        inputData = json.load(jsonInput)
        print(inputData)
        print("config file: ", inputData["config"])

    loopConfigFile=importlib.import_module(inputData["config"])
    print(loopConfigFile.config)

#--adding end bound for x
    #x=inputData["x"]
    #diff=x[-1]-x[-2]
    #x.append(x[-1]+diff)
    #x=array.array("f",x)
    xFixed=[169.0, 180.0, 191.0, 203.0, 216.0, 229.0, 243.0, 257.0, 272.0, 287.0, 303.0, 319.0, 335.0, 352.0, 369.0, 387.0, 405.0, 424.0, 443.0, 462.0, 482.0, 502.0, 523.0, 544.0, 566.0, 588.0, 611.0, 634.0, 657.0, 681.0, 705.0, 730.0, 755.0, 781.0, 807.0, 834.0, 861.0, 889.0, 917.0, 946.0, 976.0, 1006.0, 1037.0, 1068.0, 1100.0, 1133.0, 1166.0, 1200.0, 1234.0, 1269.0, 1305.0, 1341.0, 1378.0, 1416.0, 1454.0, 1493.0, 1533.0, 1573.0, 1614.0, 1656.0, 1698.0, 1741.0, 1785.0, 1830.0, 1875.0,
            1921.0, 1968.0, 2016.0, 2065.0, 2114.0, 2164.0, 2215.0, 2267.0, 2320.0, 2374.0, 2429.0, 2485.0, 2542.0, 2600.0, 2659.0, 2719.0, 2780.0, 2842.0, 2905.0, 2969.0, 3034.0, 3100.0, 3167.0, 3235.0, 3305.0, 3376.0, 3448.0, 3521.0, 3596.0, 3672.0, 3749.0, 3827.0, 3907.0, 3988.0, 4070.0, 4154.0, 4239.0, 4326.0, 4414.0, 4504.0, 4595.0, 4688.0, 4782.0, 4878.0, 4975.0, 5074.0, 5175.0, 5277.0, 5381.0, 5487.0, 5595.0, 5705.0, 5817.0, 5931.0, 6047.0, 6165.0, 6285.0, 6407.0, 6531.0, 6658.0,
            6787.0, 6918.0, 7052.0, 7188.0, 7326.0, 7467.0, 7610.0, 7756.0, 7904.0, 8055.0, 8208.0, 8364.0, 8523.0, 8685.0, 8850.0, 9019.0, 9191.0, 9366.0, 9544.0, 9726.0, 9911.0, 10100.0, 10292.0, 10488.0, 10688.0, 10892.0, 11100.0, 11312.0, 11528.0, 11748.0, 11972.0, 12200.0, 12432.0, 12669.0, 12910.0, 13156.0]
    xFixed=array.array("f", xFixed)
    xCenter=inputData["x"]
    xEdge=[]
    xDiff=0
    for i in range(len(xCenter)-1):
        xDiff=(xCenter[i+1]-xCenter[i])/2
        xEdge.append(xCenter[i]-xDiff)

    xEdge.append(xCenter[len(xCenter)-1]+xDiff)
    x=array.array("f",xEdge)
    print("xedge: ", x)



    y=inputData["y"]
    y=array.array("f",y)
    #---- Size check
    if len(x) !=len(y):
        print "x and y file have array of different sizes!"
        return -1

    rootFile=TFile(outDir+rootName+".root", "RECREATE")
    bkgFitHist=TH1F(histName, histName, len(xFixed)-1, xFixed)
    yMod=[]
    count=0
    for i in range(len(xFixed)):
        if xEdge[0]>xFixed[i] or xEdge[-1]<xFixed[i]:
            print "xFixed", xFixed[i]
            print "xEdge0", xEdge[0]
            print "xEdge-1", xEdge[-1]
            print "saving y to be 0"
            print "----------------"
            bkgFitHist.SetBinContent(i+1, 0)
            yMod.append(0.)
            count=count+1
        else :
            bkgFitHist.SetBinContent(i+1, y[i-count])
            yMod.append(y[i-count])
            print "xFixed", xFixed[i]
            print "xEdge" , xEdge[i-count]
            print "saving y to be valued"
            print "----------------"


#testing the above is doing the right thing
    for i in range(len(xFixed)):
        if yMod>0.0:
            print("yMod",yMod)
            print("first x edge that has a a non zero value is ", xFixed[i])
            break


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

def loopSaveTxtRoot(loopConfig, inputDir, inputTxtTemplate, outDir):
    """looping through the loop.py info for the file"""
    #load loop config
    loopConfigFile=importlib.import_module(loopConfig)
    # find title root name TODO remove this. this is not necessary
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
                        saveTxtToRoot(inputDir, txtFileName,outDir, txtFileName, "bkgEstFromGP")
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
    #TODO argParse inputFileTemplate
    template={"inputTxtDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/outputforPython2ToRoot/",
            #"inputFile":"output_testprintTXT_Ph100_mR750_gSM2_Mul10.txt",
            "inputFile":"output_testFixedDataBkg_Ph100_mR950_gSM3_Mul5.txt",
            #"inputFile":"output_testFixedData_Ph100_mR500_gSM3_Mul5.txt",
            "outDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/outputRootForSetLimitOneMassPoint/"}

    with open(template["inputTxtDir"]+template["inputFile"]) as jsonInput:
        inputData = json.load(jsonInput)
        print(inputData)
        print("config file: ", inputData["config"])

#----config for loop made, run loop
    loopSaveTxtRoot(inputData["config"], template["inputTxtDir"], template["inputFile"], template["outDir"])
  #  saveTxtToRoot(config['inputTxtDir'],config['inputFile'], "something.root", "hist")
