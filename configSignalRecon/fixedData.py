config={"title":"newCustomFixedData",
        "trial":15,
        "xMin":300,
        "xMax":1500,
        #-------Bkg Data file
        "bkgDataFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/MC_Dan.h5",
        "bkgDataFileTDir":"dijetgamma_g85_2j65",
        "bkgDataFileHist":"Zprime_mjj_var",
        #------Fixed Data File
        "fixedDataDir":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/MC/",
        "fixedDataFile":"MC_bkgndNSig_dijetgamma_g85_2j65_Ph100_ZPrimemRp5_gSM0p3_mulX10.h5",
        "fixedDataTDir":"",
        "fixedDataFileHist":"",
        #-------signal reconstruction
        "mass": 500, #GeV
        #------signal template
        "sigTemplateFile":"/lustre/SCRATCH/atlas/ywng/WorkSpace/r21/gp-toys/data/all/signal/reweighted_Signal_dijetgamma_g85_2j65_36p1fb.h5",
        "sigTemplateHist":"reweighted_Ph100_ZPrimemR500_gSM0p3",
        #-------output
        "outputDir":"./results",
#-------------------------place holder
        "officialFitFile":"data/all/Step1_SearchPhase_Zprime_mjj_var.h5"}
config['sigBkgDataFile']=config["fixedDataFile"]

