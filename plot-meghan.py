#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp
from iminuit import Minuit
import scipy.special as ssp


def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    return parser.parse_args()

FIT3_PARS = ['p0','p1','p2']
FIT4_PARS = ['p0', 'p1', 'p2','p3']

def dataCut(xMin, xMax, yMin, x, y, xerr, yerr):
    valid_x = (x > xMin) & (x < xMax)
    valid_y = y > yMin
    valid = valid_x & valid_y
    x, y = x[valid], y[valid]
    xerr, yerr = xerr[valid], yerr[valid]
    return x, y, xerr, yerr

def getDataPoints(fileName, tDirName, histName):
    with File(fileName,'r') as h5file:
        x, y, xerr, yerr = get_xy_pts(h5file[tDirName][histName])
        return x, y, xerr, yerr

def significance (xEst, xData, cov_x, yerr):
    return (xData-xEst) / np.sqrt(np.diag(cov_x) + yerr**2)

class Mean():
    def __init__(self, params):
        self.p0=params[0]
        self.p1=params[1]
        self.p2=params[2]
    def get_value(self, t, xErr=[]):
        sqrts = 13000.
        p0, p1, p2 = self.p0, self.p1, self.p2
        # steps = np.append(np.diff(t), np.diff(t)[-1])
        # print(steps)
        print("size of t",t.shape)
        #print("size of xErr", xErr.shape)
        vals = (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))
        #print(vals)
        return vals

def get_kernel(Amp, decay, length, power, sub):
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)

def get_kernel_Xtreme(Amp, decay, length, power, sub, mass, tau):
    kernelsig = A * ExpSquaredCenteredKernel(m=mass, t=tau)
    kernelbkg = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
    return kernelsig+kernelbkg

def get_xy_pts(group):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values'])
    edges = np.asarray(group['edges'])
    errors = np.asarray(group['errors'])
    center = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    return center, vals[1:-1], widths, errors[1:-1]
def run_vs_4paramFit():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')
    xSig, ySig, xerrSig, yerrSig = getDataPoints(args.signal_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    x, y, xerr, yerr = dataCut(0, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xMinFit=300
    xFit, yFit, xerrFit, yerrFit= dataCut(xMinFit, 1500, 0, xRaw, yRaw,xerrRaw,yerrRaw)

    # make an evently spaced x
    t = np.linspace(np.min(x), np.max(x), 500)

    #calculating the log-likihood and minimizing for the gp
    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(20, lnProb)
    fit_pars = [best_fit[x] for x in FIT3_PARS]
    #making the GP
    kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    #making the kernel
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)


    ##----4 param fit function
    lnProb = logLike_4ff(xFit,yFit,xerrFit)
    minimumLLH, best_fit_params = fit_4ff(100, lnProb)
    fit_mean = model_4param(xFit, best_fit_params, xerrFit)

    #calculating significance
    signif = significance(mu_x, y, cov_x, yerr)

    initialCutPos=np.argmax(x>xMinFit)

    sigFit =significance(fit_mean, y[initialCutPos:], cov_x[initialCutPos:], yerr[initialCutPos:])
    std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    with Canvas(f'compareGPvs4Param{ext}','GP vs 4 param fit') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        can.ax.plot(xFit, fit_mean, '-g')
        can.ax.fill_between(t, mu - std, mu + std,
                            facecolor=(0, 1, 0, 0.5),
                            zorder=5, label='err = 1')
        can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio2.stem(xFit, sigFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)

def run_vs_3paramFit():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')
    xSig, ySig, xerrSig, yerrSig = getDataPoints(args.signal_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    x, y, xerr, yerr = dataCut(0, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xMinFit=300
    xFit, yFit, xerrFit, yerrFit= dataCut(xMinFit, 1500, 0, xRaw, yRaw,xerrRaw,yerrRaw)

    # make an evently spaced x
    t = np.linspace(np.min(x), np.max(x), 500)

    #calculating the log-likihood and minimizing for the gp
    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(20, lnProb)

    fit_pars = [best_fit[x] for x in FIT3_PARS]

    #making the GP
    kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    #making the kernel
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    # calculating the fit function value

    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]
    print ("best param meghan GP minmization:", best)
    meanFromGPFit=Mean(best)

    fit_meanM =meanFromGPFit.get_value(x,xerr) #so this is currently shit. can't get the xErr thing working
    print("fit_meanM:",fit_meanM)

    #fit_mean_smooth = gp_new.mean.get_value(t)

    #----3 param fit function in a different way
    lnProb = logLike_3ff(xFit,yFit,xerrFit)
    minimumLLH, best_fit_params = fit_3ff(100, lnProb)
    fit_mean = model_3param(xFit, best_fit_params, xerrFit)

    ##----4 param fit function
    #lnProb = logLike_4ff(xFit,yFit,xerrFit)
    #minimumLLH, best_fit_params = fit_4ff(100, lnProb)
    #fit_mean = model_4param(xFit, best_fit_params, xerrFit)


    #calculating significance 
    signif = significance(mu_x, y, cov_x, yerr)

    initialCutPos=np.argmax(x>xMinFit)

    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    sigFit =significance(fit_mean, y[initialCutPos:], cov_x[initialCutPos:], yerr[initialCutPos:])
    #sigit = (mu_x)
    std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    title="compareGPvs3Param"
    with Canvas(f'%s{ext}'%title) as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        can.ax.plot(xFit, fit_mean, '-b')
        #this only works with the xErr part of Mean commented out
        #can.ax.plot(x, fit_meanM, '.g')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        #can.ax.fill_between(t, mu - std, mu + std,
                            #facecolor=(0, 1, 0, 0.5),
                            #zorder=5, label='err = 1')
        #can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.stem(xFit, sigFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.stem(x, signif, markerfmt='.', basefmt=' ')
        can.save(title)
def run_vs_UA2Fit():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points/fit
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g150_2j25', 'Zprime_mjj_var')
    xSig, ySig, xerrSig, yerrSig = getDataPoints(args.signal_file, 'dijetgamma_g150_2j25', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    x, y, xerr, yerr = dataCut(0, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xMinFit=300
    xFit, yFit, xerrFit, yerrFit= dataCut(xMinFit, 1500, 0, xRaw, yRaw,xerrRaw,yerrRaw)

    # make an evently spaced x
    t = np.linspace(np.min(x), np.max(x), 500)

    #calculating the log-likihood and minimizing for the gp
    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(100, lnProb)

    fit_pars = [best_fit[x] for x in FIT3_PARS]

    #making the GP
    kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    #making the kernel
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    # calculating the fit function value

    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]
    print ("best param meghan GP minmization:", best)
    meanFromGPFit=Mean(best)

    fit_meanM =meanFromGPFit.get_value(x,xerr) #so this is currently shit. can't get the xErr thing working
    print("fit_meanM:",fit_meanM)

    #fit_mean_smooth = gp_new.mean.get_value(t)

    #----3 param fit function in a different way
    lnProbUA2 = logLike_UA2(xFit,yFit,xerrFit)
    minimumLLH, best_fit_params = fit_UA2(1000, lnProbUA2)
    fit_mean = model_UA2(xFit, best_fit_params, xerrFit)

    ##----4 param fit function
    #lnProb = logLike_4ff(xFit,yFit,xerrFit)
    #minimumLLH, best_fit_params = fit_4ff(100, lnProb)
    #fit_mean = model_4param(xFit, best_fit_params, xerrFit)


    #calculating significance
    signif = significance(mu_x, y, cov_x, yerr)

    initialCutPos=np.argmax(x>xMinFit)

    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    sigFit =significance(fit_mean, y[initialCutPos:], cov_x[initialCutPos:], yerr[initialCutPos:])
    #sigit = (mu_x)
    #std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    title="compareGPvs3UA2New"
    with Canvas(f'%s{ext}'%title) as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        can.ax.plot(xFit, fit_mean, '-b')
        #this only works with the xErr part of Mean commented out
        #can.ax.plot(x, fit_meanM, '.g')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        #can.ax.fill_between(t, mu - std, mu + std,
                            #facecolor=(0, 1, 0, 0.5),
                            #zorder=5, label='err = 1')
        #can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.stem(xFit, sigFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio2.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio2.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)

def run_vs_SearchPhase_UA2Fit():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    x, y, xerr, yerr = dataCut(0, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xMinFit=303
    xMaxFit=1500
    xFit, yFit, xerrFit, yerrFit= dataCut(xMinFit, xMaxFit, 0, xRaw, yRaw,xerrRaw,yerrRaw)

    # make an evently spaced x
    t = np.linspace(np.min(x), np.max(x), 500)

    #calculating the log-likihood and minimizing for the gp
    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(20, lnProb)

    fit_pars = [best_fit[x] for x in FIT3_PARS]

    #making the GP
    kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    #making the kernel
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    # calculating the fit function value

    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]
    print ("best param meghan GP minmization:", best)
    meanFromGPFit=Mean(best)

    fit_meanM =meanFromGPFit.get_value(x,xerr) #so this is currently shit. can't get the xErr thing working
    print("fit_meanM:",fit_meanM)

    #fit results from search phase
    best_fit_params= (1.0779, 2.04035, 58.5769, -157.945)
    fit_mean = model_UA2(xFit, best_fit_params, xerrFit)

    ##----4 param fit function
    #lnProb = logLike_4ff(xFit,yFit,xerrFit)
    #minimumLLH, best_fit_params = fit_4ff(100, lnProb)
    #fit_mean = model_4param(xFit, best_fit_params, xerrFit)


    #calculating significance
    signif = significance(mu_x, y, cov_x, yerr)

    initialCutPos=np.argmax(x>xMinFit)
    finalCutPos=np.argmin(x<xMaxFit)

    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    sigFit =significance(fit_mean, y[initialCutPos:], cov_x[initialCutPos:], yerr[initialCutPos:])
    #sigit = (mu_x)
    #std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    title="compareGPvsSearchPhaseUA2"
    with Canvas(f'%s{ext}'%title) as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        can.ax.plot(xFit, fit_mean, '-b')
        #this only works with the xErr part of Mean commented out
        #can.ax.plot(x, fit_meanM, '.g')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        #can.ax.fill_between(t, mu - std, mu + std,
                            #facecolor=(0, 1, 0, 0.5),
                            #zorder=5, label='err = 1')
        #can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.stem(xFit, sigFit, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.save(title)
def run_bkgndVswithSig_GP():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    mass = 750
    # Getting data points
    xRaw, yRaw, xerrRaw, yerrRaw = getDataPoints(args.input_file, 'dijetgamma_g85_2j65', 'Zprime_mjj_var')
    xSig, ySig, xerrSig, yerrSig = getDataPoints(args.signal_file, 'dijetgamma_g150_2j25', 'Zprime_mjj_var')

    #Data processing, cutting out the not desired range
    xBkg, yBkg, xerrBkg, yerrBkg = dataCut(0, 1500, 0, xRaw, yRaw, xerrRaw, yerrRaw)
    xSig, ySig, xerrSig, yerrSig = dataCut(300,1500, 0, xSig, ySig, xerrSig, yerrSig)

    # make an evently spaced x
    t = np.linspace(np.min(xBkg), np.max(xBkg), 500)

    #Drawing the raw data of just the bkgnd and Sig + bkgnd
    with Canvas(f'compareRawBkgndAndBkgndWithSig{ext}') as can:
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g')
        can.ax.errorbar(xSig, ySig, yerr=yerrSig, fmt='.r')
        can.ax.set_yscale('log')

    #bkgnd calculating the log-likihood and minimizing for the gp
    lnProbBkg = logLike_minuit(xBkg, yBkg, xerrBkg)
    min_likelihood, best_fit = fit_gp_minuit(20, lnProbBkg)

    fit_pars = [best_fit[xBkg] for xBkg in FIT3_PARS]

    #making the GP
    kargs = {xBkg:yBkg for xBkg, yBkg in best_fit.items() if xBkg not in FIT3_PARS}
    kernel_Bkg = get_kernel(**kargs)
    print(kernel_Bkg.get_parameter_names())
    #making the kernel
    gp_Bkg = george.GP(kernel_Bkg, mean=Mean(fit_pars), fit_mean = True)
    gp_Bkg.compute(xBkg, yerrBkg)
    muBkg, covBkg = gp_Bkg.predict(yBkg, t)
    mu_xBkg, cov_xBkg = gp_Bkg.predict(yBkg, xBkg)

    #signal calulating the log_likihood and minizming for the GP
    # calculating the fit function value
    lnProbSig = logLike_gp_fitgpsig(xSig, ySig, xerrSig)
    min_likelihoodSig, best_fitSig = fit_gp_fitgpsig_minuit(20, lnProbSig)

    fit_pars = [best_fit[xSig] for xSig in FIT3_PARS]

    #making the GP
    #print("kargs",type(kargs))
    #kargs["mass"]=mass
    #kargs = {xSig:ySig for xSig, ySig in best_fit.items() if xSig not in FIT3_PARS}
    Args=kargs+best_fitSig
    kernel_Sig = get_kernelXtreme(**Arg)
    print(kernel_Sig.get_parameter_names())
    #making the kernel
    gp_sig = george.GP(kernel_Sig, mean=Mean(fit_pars), fit_mean = True)
    gp_sig.compute(xSig, yerrSig)
    muSig, covSig = gp_sig.predict(ySig, t)
    mu_xSig, cov_xSig = gp_Sig.predict(ySig, xSig)
    #GP compute minimizes the log likelihood of the best_fit function
    best = [best_fit[x] for x in FIT3_PARS]



    #calculating significance
    signBkg = significance(mu_xBkg, yBkg, cov_xBkg, yerrBkg)


    #sigFit = (fit_mean - y[initialCutPos:]) / np.sqrt(np.diag(cov_x[initialCutPos:]) + yerr[initialCutPos:]**2)
    signSig =significance(mu_xSig, ySig, cov_xSig, yerrSig)
    #sigit = (mu_x)
    std = np.sqrt(np.diag(cov_xBkg))

    ext = args.output_file_extension
    with Canvas(f'compareGPvs3Param{ext}') as can:
        can.ax.errorbar(xBkg, yBkg, yerr=yerrBkg, fmt='.g')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(xBkg, muSig, '-g')
        can.ax.plot(xSig, muSig, '-r')
        #this only works with the xErr part of Mean commented out
        #can.ax.plot(x, fit_meanM, '.g')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        #can.ax.fill_between(t, mubkg - std, mu + std,
                            #facecolor=(0, 1, 0, 0.5),
                            #zorder=5, label='err = 1')
        can.ratio.stem(xBkg, signBkg, markerfmt='.', basefmt=' ')
        can.ratio2.stem(xSig, signSig, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)

# _________________________________________________________________
# stuff copied from Meghan

class logLike_minuit:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2):
        kernel = get_kernel(Amp, decay, length, power, sub)
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return np.inf

def fit_gp_minuit(num, lnprob):

    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0)
    guesses = {
        'amp': 5701461179.0,
        'p0': 52.465,
        'p1': 53.66,
        'p2': -1.199
    }
    def bound(par, neg=False):
        if neg:
            return (-2.0*guesses[par], 2.0*guesses[par])
        return (guesses[par] * 0.5, guesses[par] * 2.0)
    for i in range(num):
        iamp = np.random.random() * 2*guesses['amp']
        idecay = np.random.random() * 0.64
        ilength = np.random.random() * 5e7
        ipower = np.random.random() * 1.0
        isub = np.random.random() * 1.0
        #ip0 = np.random.random() * guesses['p0'] * 2
        #ip1 = np.random.random() * guesses['p1'] * 2
        #ip2 = np.random.random() * guesses['p2'] * 2
        ip0 = guesses['p0']
        ip1 = guesses['p1']
        ip2 = guesses['p2']
        m = Minuit(lnprob, throw_nan = True, pedantic = True,
                   print_level = 0, errordef = 0.5,
                   Amp = iamp,
                   decay = idecay,
                   length = ilength,
                   power = ipower,
                   sub = isub,
                   p0 = ip0, p1 = ip1, p2 = ip2,
                   error_Amp = 1e1,
                   error_decay = 1e1,
                   error_length = 1e1,
                   error_power = 1e1,
                   error_sub = 1e1,
                   error_p0 = 1e-2, error_p1 = 1e-2, error_p2 = 1e-2,
                   limit_Amp = bound('amp'),
                   limit_decay = (0.01, 1000),
                   limit_length = (100, 1e8),
                   limit_power = (0.01, 1000),
                   limit_sub = (0.01, 1e6),
                   #limit_p0 = bound('p0', neg=False),
                   #limit_p1 = bound('p1', neg=True),
                   #limit_p2 = bound('p2', neg=True))
                   limit_p0 = (0,100),
                   limit_p1 = (-100,100),
                   limit_p2 = (-100,100))
        m.migrad()
        print(m.fval)
        if m.fval < min_likelihood and m.fval != 0.0:
            min_likelihood = m.fval
            best_fit_params = m.values
    print("min LL", min_likelihood)
    print(f'best fit params {best_fit_params}')
    return min_likelihood, best_fit_params

#------GP signal plus bkgnd
class logLike_gp_fitgpsig:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr

    def __call__(self, A, mass, tau):
        Amp, decay, length, power, sub, p0, p1, p2 = fixedHyperparams
        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
        kernel2 = A * ExpSquaredCenteredKernel(m=mass, t=tau)
        kernel = kernel1 + kernel2
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y - model_gp((p0, p1, p2), self.x, self.xerr))
        except:
            return np.inf


def fit_gp_fitgpsig_minuit(lnprob, Print=True):
    bestval = np.inf
    bestargs = (0, 0, 0)
    passedFit = False
    numRetries = 0

    for i in range(100):
        init0 = np.random.random() * 500.
        init1 = np.random.random() * 4000.
        init2 = np.random.random() * 200.
        m = Minuit(lnprob, throw_nan=False, pedantic=False, print_level=0, errordef=0.5,
                   A=init0, mass=init1, tau=init2,
                   error_A=1., error_mass=1., error_tau=1.,
                   limit_A=(1, 1e5), limit_mass=(1000, 7000), limit_tau=(100, 500))
        fit = m.migrad()
        if m.fval < bestval:
            bestval = m.fval
            bestargs = m.args

    if Print:
        print("min LL", bestval)
        print ("best fit vals", bestargs)
    return bestval, bestargs
#------fit 3
def simpleLogPoisson(x, par):
    if x < 0: 
        return np.inf
    elif (x == 0): return -1.*par
    else:
        lnpoisson = x*np.log(par)-par-ssp.gammaln(x+1.)
        return lnpoisson

def model_3param(t, params, xErr): 
    p0, p1, p2 = params
    sqrts = 13000.
    return (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2)*(xErr) )

class logLike_3ff:
    def __init__(self, x, y, xe):
        self.x = x
        self.y = y
        self.xe = xe
    def __call__(self, p0, p1, p2):
        params = p0, p1, p2
        bkgFunc = model_3param(self.x, params, self.xe)       
        logL = 0
        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            logL += -simpleLogPoisson(data, bkg)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_3ff(num,lnprob, Print=True):
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)
    for i in range(num):
        init0 = np.random.random() * 52.
        init1 = np.random.random() * 53.
        init2 = np.random.random() * -1.199
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = init0, p1 = init1, p2 = init2,
                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1, 
                  limit_p0 = (0, 100.), limit_p1 = (-100., 100.), limit_p2 = (-100., 100.))
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args 
    if Print:
        print ("min LL", minLLH)
        print ("best fit vals", best_fit_params)
    return minLLH, best_fit_params

#-------UA2
def model_UA2(t, params, xErr):
    p0, p1, p2, p3 = params
    sqrts = 13000.
    return p0 * (t/sqrts)**p1 * np.exp(-p2* (t/sqrts)-p3 *(t/sqrts)**2) *xErr

class logLike_UA2:
    def __init__(self, x, y, xe):
        self.x = x
        self.y = y
        self.xe = xe
    def __call__(self, p0, p1, p2, p3):
        params = p0, p1, p2, p3
        bkgFunc = model_UA2(self.x, params, self.xe)
        logL = 0
        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            logL += -simpleLogPoisson(data, bkg)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_UA2(num,lnprob, Print=True):
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)
    for i in range(num):
        init0 = np.random.random() * 52.
        init1 = np.random.random() * 53.
        init2 = np.random.random() * -1.199
        init3 = np.random.random() * 1
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = init0, p1 = init1, p2 = init2, p3= init3,
                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1, error_p3 = 1e-1,
                  limit_p0 = (0, 100.), limit_p1 = (-100., 100.), limit_p2 = (-100., 100.), limit_p3=(-100.,100.))
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args
    if Print:
        print ("min LL", minLLH)
        print ("best fit vals", best_fit_params)
    return minLLH, best_fit_params
#-------fit 4

def model_4param(t, params, xErr):
    p0, p1, p2, p3 = params
    sqrts = 13000.
    return (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(-p2-p3-np.log(t/sqrts))*(xErr) )

class logLike_4ff: #if you want to minimize, use this to calculate the log likelihood
    def __init__(self, x, y, xe):
        self.x = x
        self.y = y
        self.xe = xe
    def __call__(self, p0, p1, p2, p3):
        params = p0, p1, p2, p3
        bkgFunc = model_4param(self.x, params, self.xe)
        logL = 0
        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            logL += -simpleLogPoisson(data, bkg)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_4ff(num, lnprob, Print=True): #use this to minimize for the best fit function
            minLLH = np.inf
            best_fit_params = (0., 0., 0.)
            for i in range(num):
                init0 = np.random.random() * 52.
                init1 = np.random.random() * 53.
                init2 = np.random.random() * -1.199
                init3 = np.random.random() * 1
                m = Minuit(lnprob, throw_nan=False, pedantic=False, print_level=0,
                           p0=init0, p1=init1, p2=init2, p3=init3,
                           error_p0=1e-2, error_p1=1e-1, error_p2=1e-1, error_p3=1e-1,
                           limit_p0=(0, 100.), limit_p1=(-100., 100.), limit_p2=(-100., 100.),limit_p3=(-100.,100.))
                m.migrad()
                if m.fval < minLLH:
                    minLLH = m.fval
                    best_fit_params = m.args
            if Print:
                print ("min LL", minLLH)
                print ("best fit vals", best_fit_params)
            return minLLH, best_fit_params
#-------GP

if __name__ == '__main__':
    run_vs_3paramFit()
    #run_vs_UA2Fit()
    #run_vs_4paramFit()
    #run_bkgndVswithSig_GP()
    #run_vs_SearchPhase_UA2Fit()
    #run_vs_UA2Fit()
