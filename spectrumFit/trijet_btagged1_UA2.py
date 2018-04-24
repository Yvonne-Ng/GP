from runFunctions import spectrumGlobalFit

if __name__=="__main__":
#-----------a template config file -------#
    config={#-----Title
            "title": "TrijetBtagged1",
            "useScaled": False,
            #-----fit range
            "xMinFit": 300,
            "xMaxFit": 1500,
            "xMinGP": 300,
            "xMaxGP": 1500,
            #-----Spectrum file input
            "dataFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/btagged/jan2018/trijet_HLT_j380_nbtag1.h5",
            "dataFileTDir": "",
            "dataFileHist": "background_mjj_var",
            #------put some placeholder file here
            "officialFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
            #-----Fit function
            "fitFunction": 0, #0: UA2; 1: 4 params
            #initial parameter for fitting
            "initParam": [0.5145306090125814, -2.1711845084356156, 37.26540673784112, -66],
            #the range of the parameter value within which it is throwing from
            "initFitParam": [100,10,100,300], #None(default): (9.6, -1.67, 56.87,-75.877 )
            # the allowed range of variable values
            "initRange": [(2000000, 50000000.),(-10, 10),(-100, 600.),(-300, 300.)] } #None(default): [(-100000, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)]
    spectrumGlobalFit.spectrumGlobalFit(config)


