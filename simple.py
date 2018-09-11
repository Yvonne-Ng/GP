from Libplotmeghan import *
from classFitFunction import *
from classData import *
from classSignalData import *

from signalReconstruct import *

from pathlib import Path

from array import *
import json
#from Root import TFile, TH1F
import os.path

#--- Setting a config ---#
config={"title":"trijetTest",
        "trial":30,
        "xMin":300,
        "xMax":1500,
        #-------bkg only data file
        "bkgDataFile":"data/dijetISRData/trijet_ystar0p75_inclusive.root.h5",
        "bkgDataFileTDir":"",
        "dataFileHist":"background_mjj_var",
        #-------signal +bkg data file
        "mass": 500, #GeV
        "testDir":"./",
        "testDataFile":"data/dijetisrdata/trijet_ystar0p75_inclusive.root.h5",
        #------signal template
        "sigTemplateFile":"data/signalTemplate/Gauss_mass850_width7.h5",
        "sigTemplateHist": "mjj_Gauss_sig_850_smooth",

        #-------output
        "outputDir":"./trijetTest",
        "officialFitFile":"data/all/Step1_SearchPhase_Zprime_mjj_var.h5"
        #------adding signal in here
        }

# create a test data set  and set some basic quantities for the fits to follow
testDataSet=dataSet(config['xMin'], config['xMax'], config['xMin'], config['xMax'], dataFile=config[
    "testDir"]+config['testDataFile'],dataFileHist=config["dataFileHist"] ,officialFitFile=config['officialFitFile'])
# to add: fit a bkg and use those fit parameters for fit all
# method 1: GP bkg kernel fit

bkgparam1, minlike1=testDataSet.gPBkgKernelFit()
print("bkg kernel params", bkgparam)
# method 2: parametric background fit
        #bkgparam2, minlike2=testDataSet.gPSigPlusBkgKernelFit()

# method 3: reconstructed method
        #search for the
# method 4: hack from signal template



# to add: set a signal template?
signalInjectedBkgndData.fitAll(trialAll=config['trial'],mass=config['mass'], bkgDataParams=())
signalReconstruction(config)

# to add drawing:
        # data, [bkg reconstructed], [signal reconstructed], [legengd]
        # draw 2 things in 1 function. bkg fit, signal reconstructed for various mass points
