#import signalReconstruction
#from runFunctions.signalReconstruct import signalReconstruction

config={"title":"NewCustomSignalLogLikelihood",
        "trial":20,
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
        "sigTemplateFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/signal/reweighted_Signal_dijetgamma_g85_2j65_36p1fb.h5",
        "sigTemplateHist":"reweighted_Ph100_ZPrimemR500_gSM0p3",

        #-------output
        "outputDir":"./results",
#-------------------------place holder
        "officialFitFile":"data/all/Step1_SearchPhase_Zprime_mjj_var.h5"}

if __name__=="__main__":
        #signalReconstruction(config)
        pass
