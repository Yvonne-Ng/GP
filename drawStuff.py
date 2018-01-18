from pygp.canvas import Canvas
import csv


# drawing the subtraction GP fit 
def drawSignalSubtractionFit(signalBkgDataSet, ySignalData, ySignalFit, ySignalFitSignificance, doLog=False):
#drawing the signal 
    ext = ".pdf"
    title="signal GP fit "
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, ySignalData, yerr=signalBkgDataSet.yerrData, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, ySignalFit, '-r', label="GP Sig + bkg Kernel fit  - GP bkgnd kernel fit")
        can.ax.legend(framealpha=0)
        ##
        can.ratio.stem(signalBkgDataSet.xData, ySignalFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

def drawSignalGaussianFit(signalBkgDataSet, asignalDataSet, doLog=False):
#drawing the signal 
    ext = ".pdf"
    title="signal Gaussian fit "
    with Canvas(f'%s{ext}'%title, "Gaussian Fit Sig", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, asignalDataSet.ySigData, yerr=signalBkgDataSet.yerrData, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGaussianFit, '-r', label="Signal Gaussian Fit")
        can.ax.legend(framealpha=0)
        ##
        #can.ratio.stem(signalBkgDataSet.xData, asignalDataSet.gaussianFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

def drawSignalReconstructed(signalBkgDataSet, asignalDataSet, doLog=False):
#drawing the signal with meghan's method 
    ext = args.output_file_extension
    title="signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPSBFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio.stem(xSB, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

def drawAllSignalFit(signalBkgDataSet, asignalDataSet, doLog=False, saveTxt=False):
#drawing the signal 
    ext = ".pdf"
    title="allSignalFit"
    with Canvas(f'%s{ext}'%title, "All Signal Fits", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, asignalDataSet.ySigData, yerr=signalBkgDataSet.yerrData, fmt='.k', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,1000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGaussianFit, '-r', label="Signal point Gaussian Fit(injected signal)")
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGPSubtractionFit, '-g', label="Signal GP Fit subtraction")
        can.ax.plot(signalBkgDataSet.xData, signalBkgDataSet.MAP_sig, '-b', label="Signal GP reconstructed Fit")
        #can.axplot(signalBkgDataset.xData, asignalDataSet.yReconsturcted)
        can.ax.legend(framealpha=0)
        #can.ratio.stem(signalBkgDataSet.xData, asignalDataSet.gaussianFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)

def drawSignalReconstructed(signalBkgDataSet, asignalDataSet, doLog=False):
#drawing the signal with meghan's method 
    ext = args.output_file_extension
    title="signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPSBFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio.stem(xSB, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing 
        can.save(title)


def drawFitDataSet(dataSet, title, saveTxt=False, saveTxtDir=None):
    # draw the data set using diffrent fits
    ext= ".pdf"
    with Canvas(f'%s{ext}'%title, "Fit Function Official ", "GP bkgnd kernel", "GP signal+bkgnd kernel", 3) as can:
        can.ax.errorbar(dataSet.xData, dataSet.yData, yerr=dataSet.yerrData, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(dataSet.x_simpleFit, dataSet.yFit_simpleFit, '-r', label="fit function")
        can.ax.plot(dataSet.xOffFit, dataSet.yFit_officialFit, '-m', label="fit function official")
        can.ax.plot(dataSet.xData, dataSet.y_GPBkgKernelFit, '-g', label="GP bkgnd kernel") #drawing 
        can.ax.plot(dataSet.xData, dataSet.y_GPSigPlusBkgKernelFit, '-b', label="GP signal kernel") 
        if saveTxtDir:
            saveTxtDir=saveTxtDir+"/"
        if saveTxt:
            with open(saveTxtDir+"dataPoints.txt", "w") as f0:
                writer = csv.writer(f0, delimiter="\t")
                writer.writerows(zip(dataSet.xData,dataSet.yData))
                print("check: ", dataSet.xData,"   ", dataSet.yData)

            with open(saveTxtDir+"simpleFit.txt", "w") as f1:
                writer = csv.writer(f1, delimiter="\t")
                writer.writerows(zip(dataSet.x_simpleFit,dataSet.yFit_simpleFit))

            with open(saveTxtDir+"fitFunctionOfficial.txt", "w") as f2:
                writer = csv.writer(f2, delimiter="\t")
                writer.writerows(zip(dataSet.xOffFit,dataSet.yFit_officialFit))

            with open(saveTxtDir+"GPBkgndKernel.txt", "w") as f3:
                writer = csv.writer(f3, delimiter="\t")
                writer.writerows(zip(dataSet.xData,dataSet.y_GPBkgKernelFit))

            with open(saveTxtDir+"GPSigPlusBkgKernel.txt", "w") as f4:
                writer = csv.writer(f4, delimiter="\t")
                writer.writerows(zip(dataSet.xData,dataSet.y_GPSigPlusBkgKernelFit))

        can.ax.legend(framealpha=0)
        can.ratio.stem(dataSet.x_simpleFit, dataSet.significance_simpleFit, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, fitSignificance, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, testsig, markerfmt='.', basefmt=' ')
        can.ratio.set_ylabel("significance")
        can.ratio2.stem(dataSet.xData, dataSet.significance_GPBkgKernelFit, markerfmt='.', basefmt=' ')
        can.ratio2.set_ylabel("significance")
        can.ratio3.set_ylabel("significance")
        can.ratio3.stem(dataSet.xData, dataSet.significance_GPSigPlusBkgKernelFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.ratio3.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

def drawChi2ToySet(dataSet, ToyDataSet, title):
    pass



