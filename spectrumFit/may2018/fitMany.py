from runFunctions import spectrumGlobalFit
import os

def spectrumType(h5Name):
    print("h5Name: ", h5Name)
    spectrumTypeKeywords=[("trijet_inclusive", ["trijet", "inclusive"]),
                          ("trijet_2btagged", ["trijet", "nbtag2"]),
                          ("dijetgamma_single_inclusive", ["dijetgamma", "inclusive", "single"]),
                          ("dijetgamma_single_2btagged", ["dijetgamma", "nbtag2", "single"]),

                          ("dijetgamma_compound_inclusive", ["dijetgamma", "inclusive", "compound"]),
                          ("dijetgamma_compound_2btagged", ["dijetgamma", "nbtag2", "compound"])]
    name=[spectrumName for spectrumName, keywords in spectrumTypeKeywords if all(keyword in h5Name for keyword in keywords) ][0]
    #for spectrumName , keywordPars
    print("name: ", name)
    return name

def initConfig(h5Dir, h5Name, fitRange, functionCode, initParams):
    """initalize and build the config for spectrumGlobalFit"""
    SeriesName=h5Name[:-8]+"_fitRange"+str(fitRange[0])+"-"+str(fitRange[1])+"_"+spectrumGlobalFit.fitFunctionFromCode[functionCode]
    config={#-----Title
            "title": SeriesName,
            "useScaled": False, # True for MC to make data-like error
            #-----fit range
            "xMinFit": fitRange[0],
            "xMaxFit": fitRange[1],
            "xMinGP": fitRange[0],
            "xMaxGP": fitRange[1],
            #-----Spectrum file input
            "dataFile": h5Dir+"/"+h5Name,
            "dataFileTDir": "",
            "dataFileHist": "background_mjj_var",
            #------put some placeholder file here
            "officialFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
            #-----Fit function
            "fitFunction": functionCode, #0: UA2; 1: 4 params
            #initial parameter for fitting
            "initParam": initParams,
            #the range of the parameter value within which it is throwing from
            "initFitParam": [initParam*5 for initParam in initParams], #None(default): (9.6, -1.67, 56.87,-75.877 )
            #"initRange": [tuple(sorted[initParam*5, -initParam*5]) for initParam in initParams] #None(default): (9.6, -1.67, 56.87,-75.877 )
            "initRange":tuple([tuple(sorted([(initParam+2)*5, -(initParam+2)*5])) for initParam in initParams]),
            # the allowed range of variable values
            #"initRange": [(30000, 5000000.),(-10, 10),(-100, 600.),(-100, 500.)] } #None(default): [(-100000, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)]
            }
    return config
def fitMany(fitManyConfig):
    """does multiple spectrumGlobal fit for different spectrums and fit ranges and fit functions, saves the best fit params in a txt file"""
    print("got here 1")
    spectrumLogList=[]
    for h5File in os.listdir(fitManyConfig["inputH5Dir"]):

        if os.path.isfile(fitManyConfig["inputH5Dir"]+"/"+h5File):
            print("h5File: ", h5File)
            print("got here 2")
            spectrumT=spectrumType(h5File)


            print("fitRange: ", fitManyConfig["spectrumFitRanges"][spectrumT])
            for fitRange in fitManyConfig["spectrumFitRanges"][spectrumT]:
                for functionCode in fitManyConfig["functionCodes"]:
                    print("got here 3")
                    # hard coded to use the UA 2 params for now
                    initParams=fitManyConfig["initParams"][spectrumT][1]
                    #initParams=fitManyConfig[spectrumT][functionCode]
                    config=initConfig(fitManyConfig["inputH5Dir"], h5File, fitRange, functionCode, initParams)
                    spectrumLog=spectrumGlobalFit.spectrumGlobalFit(config)
                    spectrumLogList.append(spectrumLog)

    #writing to a output file
    print("got here 4")
    output_file=open(fitManyConfig["outputTxt"], "w+")
    for spectrumFitLog in spectrumLogList:
        output_file.write('\n'.join(f for f in spectrumFitLog))
    output_file.close()

if __name__=="__main__":
#### ---------File 1:
# list all the functions in the directory
    print("got here -2")

    fitManyConfig={"inputH5Dir": "data/dijetISRSearchPhasePilot/may2018/",
            "spectrumFitRanges": {"trijet_inclusive": [[169, 1500], [200, 1500], [300, 1500]],
                                  "trijet_2btagged": [[169, 1500], [200, 1500], [300,1500],[500, 1500]],
                                  "dijetgamma_single_inclusive": [[169, 1500],  [200, 1500] ,[300, 1500],[500, 1500]],

                                  "dijetgamma_single_2btagged": [[169, 1500], [200, 1500], [300, 1500],[500, 1500]],
                                  "dijetgamma_compound_inclusive": [[169, 1500], [200, 1500],  [300, 1500],[500, 1500]],
                                  "dijetgamma_compound_2btagged": [[169, 1500], [200, 1500],  [300, 1500],[500, 1500]]
                                  },
            "functionCodes": [1],
            "outputTxt": "pilotStudiesMC16adMay2018.txt",
            "initParams": {"trijet_inclusive": [
            [3872206.6966384314, -0.2517123575193576, 115.64147108047871, -99.19704085999939],    #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 3
                                        ],#STD 5
                           "trijet_2btagged": [
    [20028.30967920249, -0.3091682136768732, 103.60205088513618, 487.2162980824912] ,                         #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 3
                                        ],#STD 5
                           "dijetgamma_single_inclusive": [
            [20002.278591308175, -0.9118109883110979, 188.56795166982783, -68.44586390253173],    #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 3
                                        ],#STD 5
                           "dijetgamma_single_2btagged": [
            [20002.278591308175, -0.9118109883110979, 188.56795166982783, -68.44586390253173],     #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 4
                                        #STD 3
                                        ], #STD 5

                           "dijetgamma_compound_inclusive": [
            [20002.278591308175, -0.9118109883110979, 188.56795166982783, -68.44586390253173],    #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 3
                                        ],#STD 5
                           "dijetgamma_compound_2btagged": [
            [20002.278591308175, -0.9118109883110979, 188.56795166982783, -68.44586390253173],     #UA2
[52, 100, -1.199, 1]                    #STD 4
                                        #STD 4
                                        #STD 3
                                        ]#STD 5
            }}


    print("got here -1")
    fitMany(fitManyConfig)
    #h5List=os.path.listdir("data/dijetISRSearchPhasePilot/may2018/")
    #for h5File in h5List:
    #    configBuilding(h5File)
    #spectrumLog=spectrumGlobalFit.spectrumGlobalFit(config)
    #output_file=open("fitmany.txt", "w+")
    #output_file.write('\n'.join(f for f in spectrumLog))
    #output_file.close()



