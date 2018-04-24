import matplotlib
from pygp.canvas import Canvas
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# drawing the subtraction GP fit
def drawSignalSubtractionFit(signalBkgDataSet, ySignalData, ySignalFit, ySignalFitSignificance, doLog=False, title=""):
    title=title+"signalGPfit"
#drawing the signal
    ext = ".pdf"
    with Canvas(f'%s{ext}'%title, "GPSig-GPBkgFit Sig", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, ySignalData, yerr=signalBkgDataSet.yerrData, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, ySignalFit, '-r', label="GP Sig + bkg Kernel fit  - GP bkgnd kernel fit")
        can.ax.legend(framealpha=0)
        ##
        can.ratio[0].stem(signalBkgDataSet.xData, ySignalFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save(title)

def drawSignalGaussianFit(signalBkgDataSet, asignalDataSet, doLog=False, title=""):
#drawing the signal
    ext = ".pdf"
    title=[s + "SignalGaussianFit" for s in title]
    with Canvas(f'%s{ext}'%title, "Gaussian Fit Sig", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, asignalDataSet.ySigData, yerr=signalBkgDataSet.yerrData, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,10000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGaussianFit, '-r', label="Signal Gaussian Fit")
        can.ax.legend(framealpha=0)
        ##
        #can.ratio.stem(signalBkgDataSet.xData, asignalDataSet.gaussianFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save(title)

def drawSignalReconstructed(signalBkgDataSet, asignalDataSet, doLog=False, title=""):
#drawing the signal with meghan's method
    ext = args.output_file_extension
    title=title+"signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPSBFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio[0].stem(xSB, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save(title)

def drawAllSignalFit(signalBkgDataSet, asignalDataSet, doLog=False, saveTxt=False, title=""):
#drawing the signal
    ext = ".pdf"
    title=title+"allSignalFit"
    with Canvas(f'%s{ext}'%title, "All Signal Fits", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, asignalDataSet.ySigData, yerr=signalBkgDataSet.yerrData, fmt='.k', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,15000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGaussianFit, '-r', label="Signal point Gaussian Fit(injected signal)")
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGPSubtractionFit, '-g', label="Signal GP Fit subtraction")
        can.ax.plot(signalBkgDataSet.xData, signalBkgDataSet.MAP_sig, '-b', label="Signal GP reconstructed Fit")
        #can.axplot(signalBkgDataset.xData, asignalDataSet.yReconsturcted)
        can.ax.legend(framealpha=0)
        #can.ratio.stem(signalBkgDataSet.xData, asignalDataSet.gaussianFitSignificance, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save(title)

def drawSignalReconstructed(signalBkgDataSet, asignalDataSet, doLog=False,title=""):
#drawing the signal with meghan's method
    ext = args.output_file_extension
    title=title+"signalonlyMeghan"
    with Canvas(f'%s{ext}'%title, "GPSig-GPSBFit Sig", "", "", 2) as can:
        can.ax.errorbar(xSB, ySig, yerr=yerrSB, fmt='.g', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,6000000)
        can.ax.set_yscale('log')
        can.ax.plot(xSB, MAP_bkg, '-r', label="GP Sig Kernel Only fit")
        can.ax.legend(framealpha=0)
        can.ratio[0].stem(xSB, GPSigOnlySignificanceMeg, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save(title)


def drawFitDataSet(dataSet, title, saveTxt=False, saveTxtDir=None):
    # draw the data set using diffrent fits
    ext= ".pdf"
    with Canvas(f'%s{ext}'%title , "GP bkgnd kernel", "GP signal+bkgnd kernel", 2) as can:
        can.ax.errorbar(dataSet.xData, dataSet.yData, yerr=dataSet.yerrData, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        #can.ax.plot(dataSet.x_simpleFit, dataSet.yFit_simpleFit, '-r', label="fit function")
        #can.ax.plot(dataSet.xOffFit, dataSet.yFit_officialFit, '-m', label="fit function official")
        can.ax.plot(dataSet.xData, dataSet.y_GPBkgKernelFit, '-g', label="GP bkgnd kernel") #drawing
        #can.ax.plot(dataSet.xData, dataSet.y_GPSigPlusBkgKernelFit, '-b', label="GP signal kernel")
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
        #can.ratio.stem(dataSet.x_simpleFit, dataSet.significance_simpleFit, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, fitSignificance, markerfmt='.', basefmt=' ')
        #can.ratio.stem(xSBFit, testsig, markerfmt='.', basefmt=' ')
        #can.ratio.set_ylabel("significance")
        can.ratio[0].stem(dataSet.xData, dataSet.significance_GPBkgKernelFit, markerfmt='.', basefmt=' ')
        can.ratio[0].set_ylabel("significance")
        can.ratio[1].set_ylabel("significance")
        can.ratio[1].stem(dataSet.xData, dataSet.significance_GPSigPlusBkgKernelFit, markerfmt='.', basefmt=' ')
        can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

def drawChi2ToySet(dataSet, ToyDataSet, title):
    pass

def drawFit(xData=None, yerr=None, yData=None, yFit=None,sig=None, title=None, saveTxt=False, saveTxtDir=None):
    # draw the data set using diffrent fits
    ext= ".pdf"
    with Canvas(f'%s{ext}'%title , "UA2", "", 2) as can:
        can.ax.errorbar(xData, yData,yerr, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(xData, yFit, '-g', label="UA2Fit") #drawing
        #ratio plot:
        print("x: ", xData)
        print("sig: ", sig)
        if sig:
            can.ratio[0].stem(xData, sig, markerfmt='.', basefmt=' ')
            can.ratio[0].set_ylabel("significance")
            can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

#TODO: expansion to make yFit a list and sig a list

def drawFit2(xData=None, yerr=None, yData=None, yFit=None,yFit2=None, sig=None, signiLegend=None, title=None, saveTxt=False, saveTxtDir=None):
    # draw the data set using diffrent fits
    ext= ".pdf"
    with Canvas(f'%s{ext}'%title , "UA2", "", 2) as can:
        can.ax.errorbar(xData, yData,yerr, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        can.ax.plot(xData, yFit, '-g', label="bkgndKernelFit") #drawing
        can.ax.plot(xData, yFit2, '-r', label="GP background+signal") #drawing
        if saveTxtDir==None:
            saveTxtDir=""
        with open(saveTxtDir+"dataPoints.txt", "w") as f0:
            writer = csv.writer(f0, delimiter="\t")
            writer.writerows(zip(xData,yData))
        with open(saveTxtDir+"bkgndReconstructed.txt", "w") as f0:
            writer = csv.writer(f0, delimiter="\t")
            writer.writerows(zip(xData,yFit))
        with open(saveTxtDir+"bkgndSigReconstructed.txt", "w") as f0:
            writer = csv.writer(f0, delimiter="\t")
            writer.writerows(zip(xData,yFit2))
        can.ax.legend(framealpha=0)
        if sig:
            for i in range(len(sig)):
                print("i", i)
                can.ratio[i].stem(xData, sig[i], markerfmt='.', basefmt=' ')
                if signiLegend!=None:
                    can.ratio[i].set_ylabel(signiLegend[i])
                can.ratio[i].axhline(0, linewidth=1, alpha=0.5)
        can.save("results/"+title)

def drawFit3(xData=None, yerr=None, yData=None, yFit=None,yFit2=None, yFit3=None, legend=[],sig=None, title=None, saveTxt=False, saveTxtDir=None):
    # draw the data set using diffrent fits
    ext= ".pdf"
    with Canvas(f'%s{ext}'%title , "UA2", "", 2) as can:
        can.ax.errorbar(xData, yData,yerr, fmt='.g', label="datapoints") # drawing the points
        can.ax.set_yscale('log')
        if legend ==[]:
            legend=["bkgKernelFit","GP bakcground +signal kernel fit", "GP backgtound+signal fit -custom signal template reconsturct"]

        can.ax.plot(xData, yFit, '-g', label=legend[0]) #drawing
        can.ax.plot(xData, yFit2, '-r', label=legend[1]) #drawing
        can.ax.plot(xData, yFit3, '-b', label=legend[2]) #drawing
        #ratio plot:
        #print("x: ", xData)
        #print("sig: ", sig)
        can.ax.legend(framealpha=0)
        if sig:
            can.ratio[0].stem(xData, sig, markerfmt='.', basefmt=' ')
            can.ratio[0].set_ylabel("significance")
            can.ratio[0].axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

def makePrettyPlots_chi2(GPchi2, BKGchi2, title, drawchi2=False, xname=r'$\chi^{2}$/d.o.f.', label1 = "Gaussian Process", label2 = "Fit Function"):
    f, (ax1) = plt.subplots(1, figsize=(12,12), gridspec_kw = {'height_ratios':[1, 1]})
    f.suptitle(title, fontsize=40)

    lowx = min(min(GPchi2), min(BKGchi2))
    highx = max(max(GPchi2), max(BKGchi2))
    bins = np.linspace(lowx, highx, 100)

    hGP, _, _ =ax1.hist(GPchi2, bins=bins, alpha=0.7, color="g", label='Gaussian Process model')
    hBKG, _, _ =ax1.hist(BKGchi2, bins=bins, alpha=0.7, color='b', label="ad-hoc fit")
    ax1.tick_params(axis='y', labelsize=30)
    ax1.tick_params(axis='x', labelsize=30)
    ax1.set_xlabel(xname, fontsize=40)
    plt.xlim(0.4, 4)

    plt.legend(prop={'size':20})
    ax1.save("chi2.pdf")

def drawChi2Dist(title="",chi2DistDict=[], legend=[]):
    #differnt injection
    ext= ".pdf"
    nList=[]
    binsList=[]
    pathcesList=[]
    color=['g', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    i=0
    with Canvas("Testchi2.pdf") as can:
        for tag in legend:
            pseudoTag=tag+"Pseudo"
            print("chi2DistDict[tag]: ", chi2DistDict[pseudoTag])
            print("color: ", color[i])
            #n, bins, patches, = plt.hist(chi2DistDict[pseudoTag], 50, normed=1, facecolor=color[i], alpha=0.75, label=legend[i])
            can.ax.hist(chi2DistDict[pseudoTag], bins=20, facecolor=color[i], alpha=0.75, label=legend[i])
            can.ax.set_xlim(0,3)
            #bins = np.histogram(chi2DistDict[pseudotag], edges)

            #nList.append(n)
            #binsList.append(bins)
            #patchesList.append(patches)
            i=i+1

        can.ax.legend(framealpha=0)
        can.save("testchi2.pdf")

    #Plt.xlabel('chi2/ndf')
    #Plt.ylable('count')
    #Plt.legend(prop={'size':20})
    #plt.savefig("testchi2.pdf")


    #with Canvas(f'%s{ext}'%title , "UA2", "", 2) as can:
        #f, (ax1) = plt.subplots(1, figsize=(12,12), gridspec_kw = {'height_ratios':[1, 1]})
        #f, (ax1) = plt.subplot(211)
        #f.subtitle(title, fontsize=40)

        #lowx = min(min(GPchi2), min(BKGchi2))
        #highx = max(max(GPchi2), max(BKGchi2))
        #bins = np.linspace(lowx, highx, 100)

    #dist=[]

    #for i in range(len(chi2DistDict)):
    #    pseudoTag=legend[i]+"Pseudo"
    #    can.ax.plot(, y1)

    #    dist[i], _, _ =ax1.hist(chi2DistDict[psuedoTag], bins=bins, alpha=0.7, color=color[i], label=legend[i])
    #    w
    #    ax1.tick_params(axis='y', labelsize=30)
    #    ax1.tick_params(axis='x', labelsize=30)
    #    ax1.set_xlabel(xname, fontsize=40)
    #    plt.xlim(0.4, 4)

    #ax1.save("Test_chi2.pdf")


def drawAllSignalFitYvonne(signalBkgDataSet, asignalDataSet, doLog=False, saveTxt=False, title="",significanceSig=None, sig=None):
#drawing the signal
    ext = ".pdf"
    title=title+"allSignalFit"
    with Canvas(f'%s{ext}'%title, "All Signal Fits", "", "", 2) as can:
        can.ax.errorbar(signalBkgDataSet.xData, asignalDataSet.ySigData, yerr=signalBkgDataSet.yerrData, fmt='.k', label="signal MC injected") # drawing the points
        can.ax.set_ylim(0.1,15000.0)
        if doLog:
            can.ax.set_yscale('log')
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGaussianFit, '-r', label="Signal point Gaussian Fit(injected signal)")
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.yGPSubtractionFit, '-g', label="Signal GP Fit subtraction")
        #can.ax.plot(signalBkgDataSet.xData, asignalDataSet.sig['Gaussian'],'-m', label="Signal GP Kernel reconstructed Gaussian Fit")
        print(asignalDataSet.sig['custom'])
        can.ax.plot(signalBkgDataSet.xData, asignalDataSet.sig['custom'],'-b', label="Signal GP Kernel reconstructed signal template Fit")
        #can.ax.plot(signalBkgDataSet.xData, asignalDataSet.sig['customTest'],'-k', label="Signal GP Kernel reconstructed signal template Fit default")
        #accidentally broke gaussian signal
#print both signal
#        if sig:
#            for i in range(len(sig)):
#                print("i", i)
#                can.ratio[i].stem(signalBkgDataSet.xData, significanceSig[i], markerfmt='.', basefmt=' ')
#                can.ratio[i].set_ylabel(sig[i])
#                print("legend: ", sig[i])
#                can.ratio[i].axhline(0, linewidth=1, alpha=0.5)
#                can.ax.legend(framealpha=0)

        can.ratio[1].stem(signalBkgDataSet.xData, significanceSig[1], markerfmt='.', basefmt=' ')
        can.ratio[1].set_ylabel(sig[1])
        print("legend: ", sig[1])
        can.ratio[1].axhline(0, linewidth=1, alpha=0.5)
        can.ax.legend(framealpha=0)
            #can.ratio.stem(signalBkgDataSet.xData, asignalDataSet.gaussianFitSignificance, markerfmt='.', basefmt=' ')
            #can.ratio.axhline(0, linewidth=1, alpha=0.5)
            #can.ax.plot(xSB, ymuGP_KernBkg_SB, '-g', label="GP bkgnd kernel") #drawing
        can.save("results/"+title)
