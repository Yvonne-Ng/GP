from runFunctions import spectrumGlobalFit

if __name__=="__main__":
#-----------a template config file -------#
    config={#-----Title
            "title": "TrijetInclusiveABCD",
            "useScaled": False,
            #-----fit range
            "xMinFit": 300,
            "xMaxFit": 1500,
            "xMinGP": 300,
            "xMaxGP": 1500,
            #-----Spectrum file input
            "dataFile": "/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/reweighted_hist-background_ABCD_trijet.h5",
            "dataFileTDir": "",
            "dataFileHist": "Zprime_mjj_var",
            #------put some placeholder file here
            "officialFitFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/Step1_SearchPhase_Zprime_mjj_var.h5",
            #-----Fit function
            "fitFunction": 0, #0: UA2; 1: 4 params
            #initial parameter for fitting
            "initParam":[8230953.188903861, -1.5264892708211963, 185.77465370632643, -571.125106843567],

            #the range of the parameter value within which it is throwing from
            "initFitParam":[100000000, 50, 600, 1000],
            # the allowed range of variable values
            "initRange": [(1000000, 50000000.),(-10, 10),(-100, 600.),(-1000, 300.)] } #None(default): [(-100000, 1000000.),(-100., 100.),(-100., 100.),(-100., 100.)]
    spectrumGlobalFit.spectrumGlobalFit(config)


