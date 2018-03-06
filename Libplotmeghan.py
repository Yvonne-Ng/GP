#!/usr/bin/env python3

'''
this defines all the functions in python which are called later by the run functions
'''
from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import SignalKernel,MyDijetKernelSimp,LocalGaussianKernel, ExpSquaredCenteredKernel#, ExpSquared#, ExpSquaredCenteredKernel, ExpSquaredKernel
from george.kernels import ExpSquaredCenteredKernel

from iminuit import Minuit
import scipy.special as ssp
import inspect
import math

import random

from numpy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from drawStuff import *

FIT3_PARS = ['p0','p1','p2']
FIT4_PARS = ['p0', 'p1', 'p2','p3']
#mass reconstruction window %
massReconstructionWindow=0.1

#General Function

#def gauss(x, *p):
#    a, b, c, d = p
#    y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d

  #  return y
#def model_gp(params, t, xerr):
#    p0, p1, p2 = params
#    sqrts = 13000.
#    return (p0 * (1.-t/sqrts)**p1 * (t/sqrts)**(p2))*xerr
#
#alternate model_gp that matched more or less with Mean


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self

def model_gp(params, t, xerr):
    p0, p1, p2 = params
    sqrts = 13000.
    return (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))
def gaussian(x, amp, cen, wid):
    return amp * exp(-(x-cen)**2 /wid)

def removeZeros(xSet,ySet, xerrSet, yerrSet):
    indice_yNot0=[i for i, y in enumerate(ySet) if y!=0]
    newXSet=[]
    newYSet=[]
    newXerrSet=[]
    newYerrSet=[]
    for i in range (len(xSet)):
        if i in indice_yNot0:
            newXSet.append(xSet[i])
            newYSet.append(ySet[i])
            newXerrSet.append(xerrSet[i])
            newYerrSet.append(yerrSet[i])
    return newXSet, newYSet, newXerrSet, newYerrSet


def dataCut(xMin, xMax, yMin, x, y, xerr, yerr):
    valid_x = (x > xMin) & (x < xMax)
    valid_y = y > yMin
    valid = valid_x & valid_y
    x, y = x[valid], y[valid]
    xerr, yerr = xerr[valid], yerr[valid]
    return x, y, xerr, yerr

def getDataPoints(fileName, tDirName, histName):
    with File(fileName,'r') as h5file:
        if tDirName=="":
            x, y, xerr, yerr = get_xy_pts(h5file[histName])
        else:
            x, y, xerr, yerr = get_xy_pts(h5file[tDirName][histName])
        return x, y, xerr, yerr

def significance (xEst, xData, cov_x, yerr): #classical definition of significance
    return (xData-xEst) / np.sqrt(np.diag(cov_x) + yerr**2)

def res_significance(Data,Bkg): #residual definition of signifiance # also calculates chi2
    pvals = []
    zvals = []
    chi2 = 0
    for i, nD in enumerate(Data):
        nB = Bkg[i]
        if nD != 0:
            if nB > nD:
                pval = 1.-ssp.gammainc(nD+1.,nB)
            else:
                pval = ssp.gammainc(nD,nB)
            prob = 1-2*pval
            if prob > -1 and prob < 1:
                zval = math.sqrt(2.)*ssp.erfinv(prob)
            else:
                zval = np.inf
            if zval > 100: zval = 20
            if zval < 0: zval = 0
            if (nD < nB): zval = -zval
        else: zval = 0

        zvals.append(zval)
        if abs(nD)==0.:
            chi2+=2
        else:
            if ((nD - nB) ** 2 / abs(nD)) <2:

                chi2 += ((nD - nB) ** 2 / abs(nD))
            else :
                chi2 += 2

    zvals=np.array(zvals)
    return zvals, chi2

def sig_model(x, N=1e5, mass=2000., width=100., xErr=0.1):
    return N*(np.exp(-(x-mass)**2/2/width/width)/np.sqrt(2*np.pi)/width)*xErr

def poissonPVal(d,b):
    if (d>=b):
        answer=ssp.gammainc(d,b)
    else:
        answer=1-ssp.gammainc(d+1,b)
    return answer

def probToSigma(prob):
    if (prob<0 and prob>1):
        return "Yvonne ERROR, prob out of range in probToSigma"
    valtouse = 1-2*prob
    if (valtouse>-1 and valtouse<1) :
        return math.sqrt(2.0)*ssp.erfinv(valtouse)
    elif (valtouse==1):
        return 20
    else:
        return -20

def resSigYvonne(Data, Bkg, weight=None):
    resultResidual=[]
    chi2Sum=0
    for i in range(len(Data)):
        if weight is not None:
            d=Data[i]/weight[i]
            b=Bkg[i]/weight[i]
        else:
            d=Data[i]
            b=Bkg[i]
        chi2Sum=(d-b)**2+chi2Sum
        #do I actually need berr
        #berr
        pVal=poissonPVal(d,b)
        frac=probToSigma(pVal)
        if (float(frac)>100): # if frac is too big
            frac=20
        if (float(frac)<0):
            frac=0
        if (d<b):
            frac=frac*-1# what do you do with -frac??
        resultResidual.append(frac)
    chi2=float(chi2Sum)/len(Data)
    return resultResidual, chi2






#making Toys
def makeToys(dataList, lumi = 3.6):
      return  np.random.poisson(dataList*lumi/lumi)

#estimating the mean for GP cclculations
class Mean():
    def __init__(self, params):
        self.p0=params[0]
        self.p1=params[1]
        self.p2=params[2]
    def get_value(self, t, xErr=[]):
        sqrts = 13000.
        p0, p1, p2 = self.p0, self.p1, self.p2
        vals = (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))
        return vals

def get_kernel(Amp, decay, length, power, sub): # getting the typical background kernel
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)
#def get_kernel_SigOnly(A, mass, tau):
#    return A* LocalGaussianKernel(mass, tau)

def get_kernel_SigOnly(A, mass, tau,l):
    return A* SignalKernel(mass, tau,l)
def get_kernel_Xtreme(Amp, decay, length, power, sub, A, mass, tau,l): # getting the signal+bkgnd kernel
    kernelsig= A* SignalKernel(mass, tau,l)
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
    best_fit_params = (0, 0, 0, 0, 0,0,0,0)
    guesses = {
        'amp': 6946279097.0,
        #'p0': 0.23,
        #'p1': 0.46,
        #'p2': 0.89
        #gotta change this to the fit function value!
        'p0': 6.00,
        'p1': 19.43,
        'p2': -2.307
    }
    def bound(par, neg=False):
        if neg:
            return (-2.0*guesses[par], 2.0*guesses[par])
        return (guesses[par] * 0.5, guesses[par] * 2.0)

    for i in range(num):
        iamp = np.random.random() * 2*guesses['amp']
        idecay = np.random.random() * 0.64
        ilength = np.random.random() * 5e5
        ipower = np.random.random() * 1000
        isub = np.random.random() * 1.0
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
                   limit_length = (5000000, 1e8),
                   #limit_power = (0.0001, 100),
                   limit_power = (800, 1000),
                   limit_sub = (0.001, 1e6),
                   limit_p0 = (-100,1000000),
                   limit_p1 = (-100,100),
                   limit_p2 = (-100,100))
        m.migrad()
        print("trial #", i)
        print ("likelihood: ",m.fval)
        print("args: ", m.args)
        if m.fval < min_likelihood and m.fval != 0.0:
            min_likelihood = m.fval
            best_fit_params = m.values
    print("GP bkgnd kernel")
    print("min ll", min_likelihood)
    print(f'best fit params {best_fit_params}')
    return min_likelihood, best_fit_params

class logLike_minuit_sig:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2, A=0, mass=0, tau=0):
        #kernel = get_kernel(Amp, decay, length, power, sub)
        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
        kernel2 = A * LocalGaussianKernel(mass, tau)
        kernel = kernel1 + kernel2
        #kernel=kernel1
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return np.inf

class logLike_minuit_sigWithWindow:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2, A=0, mass=0, tau=0,l=0):
        #kernel = get_kernel(Amp, decay, length, power, sub)
        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
        kernel2 = A * SignalKernel(mass, tau, l)
        kernel = kernel1 + kernel2
        #kernel=kernel1
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return np.inf

#####################3#
#    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2, A=0, mass=0, tau=0):
#        #Amp, decay, length, power, sub, p0, p1, p2 = fixedHyperparams
#        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
#        kernel2 = A * LocalGaussianKernel(mass, tau)
#        kernel = kernel1 #+ kernel2
#        #kernel=kernel1
#        gp = george.GP(kernel)
#        try:
#            gp.compute(self.x, np.sqrt(self.y))
#            return -gp.lnlikelihood(self.y, self.xerr)
#        except:
#            return np.inf
###
def fit_gpSig_minuit(num, lnprob, mass, initParams):

    min_likelihood = np.inf
    best_fit_params= "doesn't converge"
    guesses = {
        'amp': 6946279097.0,
        #'p0': 0.23,
        #'p1': 0.46,
        #'p2': 0.89
        #gotta change this to the fit function value!
        'p0': 6.00,
        'p1': 19.43,
        'p2': -2.307
    }
    def bound(par, neg=False):
        if neg:
            return (-2.0*guesses[par], 2.0*guesses[par])
        return (guesses[par] * 0.5, guesses[par] * 2.0)

    for i in range(num):
        iamp = initParams[0]#i*1.2*np.random.random()# * 2*guesses['amp']
        idecay = initParams[1]#*1.2*np.random.random()# * 0.64
        ilength = initParams[2]#*1.2*np.random.random()# * 5e5
        ipower = initParams[3]#*1.2*np.random.random()# * 1.0
        isub = initParams[4]#*1.2*np.random.random()# * 1.0
        ip0 =initParams[5] #*1.2*np.random.random() #guesses['p0']
        ip1 = initParams[6]#*1.2*np.random.random() #guesses['p1']
        ip2 = initParams[7]#*1.2*np.random.random() #guesses['p2']
        iA = (25000000000 *np.random.random())
        itau = 50
        iMass = mass*(1+massReconstructionWindow)
        iL=(500*np.random.random())
        if ip2>0:
            factor=1.10
        else:
            factor=0.90

        m = Minuit(lnprob, throw_nan = True, pedantic = True,
                   print_level = 0, errordef = 0.5,
                   Amp = iamp,decay = idecay, length = ilength,power = ipower,sub = isub,
                   p0 = ip0, p1 = ip1, p2 = ip2,A=iA, tau=itau, mass=iMass,l=iL,
                   error_Amp=1e1,
                   error_decay = 1e1,
                   error_length = 1e1,
                   error_power = 1e1,
                   error_sub = 1e1,
                   error_p0 = 1e-2, error_p1 = 1e-2, error_p2 = 1e-2,
                   error_A = 1e-2, error_tau = 1e-2, error_mass = 1e-2,
                   error_l=1e-2,
                   limit_Amp = (initParams[0], initParams[0]*1.001),#bound('amp'),
                   limit_decay = (initParams[1], initParams[1]*1.001),
                   limit_length = (initParams[2], initParams[2]*1.001),
                   limit_power = (initParams[3], initParams[3]*1.001),
                   limit_sub = (initParams[4], initParams[4]*1.001),
                   limit_p0 = (initParams[5],initParams[5]*1.001),
                   limit_p1 = (initParams[6], initParams[6]*1.001),
                   limit_p2 = (initParams[7],initParams[7]*factor),
                   limit_A = (50000000, 2500000000000),
                   limit_tau = (1, 1000),
                   limit_mass=(mass*(1-massReconstructionWindow), mass*(1+massReconstructionWindow)),limit_l=(80,500))
        m.migrad()
        print("trial #", i)
        print ("likelihood: ",m.fval)
        print("args: ", m.args)
        if( m.fval < min_likelihood )and( m.fval != 0.0):
            min_likelihood = m.fval
            best_fit_params = m.values
    print("GP sig kernel from bkgnd")
    print("min ll", min_likelihood)
    print(f'best fit params {best_fit_params}')
    return min_likelihood, best_fit_params

#------GP signal plus bkgnd
class logLike_gp_fitgpsig:
    def __init__(self, x, y, xerr,fixedParams,  weight=None):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.fixParam=fixedParams

        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.one(self.x.size)

    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2, A=0, mass=0, tau=0):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixParam.values()
        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
        kernel2 = A * LocalGaussianKernel(mass, tau)
        kernel = kernel1 #+ kernel2
        #kernel=kernel1
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y/weight, self.xerr/weight)
        except:
            return np.inf

def fit_gp_fitgpsig_minuit(num, lnprob, mass, initParams,Print=True):
    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0,0,0,0)
    for i in range(num):
        iAmp=np.random.random() * initParams[0]
        iDecay =np.random.random() * initParams[1]
        ilength =np.random.random() *initParams[2]
        ipower =np.random.random() * initParams[3]
        isub= np.random.random() *initParams[4]
        ip0=np.random.random() *initParams[5]
        ip1=np.random.random() * initParams[6]
        ip2=np.random.random() * initParams[7]
        iA =np.random.random() *0#(np.random.random() *2)*0.000001
        imass=np.random.random() *mass*(1+massReconstructionWindow)
        itau= np.random.random() *100000000

        m = Minuit(lnprob, throw_nan=True, pedantic=True, print_level=0, errordef=0.5, Amp=iAmp, decay=iDecay, length=ilength, power=ipower, sub=isub, p0=ip0, p1=ip1, p2=ip2, A=iA, mass=imass, tau=itau, error_Amp=1e1, error_decay=1e1, error_length=1e1, error_power=1e1, error_sub=1e1, error_p0=1e-2, error_p1=1e-2, error_p2=1e-2, error_A=1., error_mass=1., error_tau=1., limit_Amp=(-100000,100000000000), limit_decay=(0.01, 1000), limit_length=(1, 1e8), limit_power=(0.001, 1000), limit_sub=(0.001,
            1e6), limit_p0=(-100,1000000.), limit_p1=(-100., 100.), limit_p2=(-100., 100), limit_A=(-0.1,0.1), limit_mass=(mass*(1-massReconstructionWindow), mass*(1-massReconstructionWindow)), limit_tau=(-500, 500))

        m.migrad()
        print("trial #", i)
        print ("likelihood: ",m.fval)
        print("args: ", m.args)
        if m.fval < min_likelihood and m.fval != 0.0:
            min_likelihood = m.fval
            best_fit_params = m.values
    print("GP signal kernel")
    print("min LL", min_likelihood)
    print ("best fit vals", best_fit_params)
    return min_likelihood, best_fit_params
#------fit 3
def simpleLogPoisson(x, par):
    if x < 0:
        return np.inf
    elif (x == 0): return -1.*par
    else:
        lnpoisson = x*np.log(par)-par-ssp.gammaln(x+1.)
        return lnpoisson

def model_3param(params,t, xErr):
    p0, p1, p2 = params
    sqrts = 13000.

    return (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2) )

class logLike_3ff:
    def __init__(self, x, y, xe,weight=None):
        self.x = x
        self.y = y
        self.xe = xe
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)
    def __call__(self, p0, p1, p2):
        params = p0, p1, p2
        bkgFunc = model_3param(params, self.x,self.xe)
        logL = 0
        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            binWeight = self.weight[ibin]
            logL += -simpleLogPoisson(data/binWeight, bkg/binWeight)
        try:
            logL
            return logL
        except:
            return np.inf

def logLike_3ffOff(x, y, xFit, yFit, xe):
    logL = 0
    for ibin in range(len(y)):
        data = y[ibin]
        bkg = yFit[ibin]
        logL += -simpleLogPoisson(data, bkg)
    try:
        logL
        return logL
    except:
        return np.inf

def fit_3ff(num,lnprob, initParam=None, initFitParam=None, initRange=None, Print=True):
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)
    if initFitParam==None:
        initFitParam=[50000000, 200, 300]
#    initParam=[289.23835739677855, 36.840272925751094, -2.6319426698603223]
    #initParam=[1072002.0144121815, 75.21331095659207, -0.6270716885884724]
    if initParam==None:
        initParam=[21446363.154059794, 89.30345184915959, 0.10256264361501621]
    for i in range(num):
        #init0 =  np.random.random() *100
        #init1 = np.random.random() * 100
        #init2 = np.random.random() * 100


        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = initParam[0], p1 = initParam[1], p2 = initParam[2],

                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1,
                  limit_p0 = (500000, 50000000.), limit_p1 = (-100., 200.), limit_p2 = (-100., 100.))
        m.migrad()
        print("LL: ", m.fval)
        initParam=[x * np.random.random() for x in initFitParam]
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args
            initParam=[x * np.random.random() for x in initFitParam]
            print("initParam: ", initParam)
    if Print:
        print("fit function")
        print ("min LL", minLLH)
        print ("best fit vals", best_fit_params)
    return minLLH, best_fit_params

#------Fixed Param for signal reconstruction
class logLike_gp_sigRecon:
    def __init__(self, x, y, xerr,yerr,fixedHyperparams, weight=None ):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        self.fixedHyperparams=fixedHyperparams
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)
    def __call__(self, A, mass, tau):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixedHyperparams.values() #best_fit_gp
        kernel1 = Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d=sub)
        kernel2 = A * ExpSquaredCenteredKernel(m = mass, t = tau)
        #kernel2 = 1 * ExpSquaredCenteredKernel(2, 3)
        kernel = kernel1+kernel2
        gp = george.GP(kernel)
        #seriously, why are we using the log likelihood of gp???
        try:
            if weight is not None:
                gp.compute(self.x, self.yerr)
            else:
                gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y/weight - model_gp((p0,p1,p2), self.x, self.xerr)/weight)
        except:
            return np.inf

class logLike_gp_sigRecon_diffLog:
    def __init__(self, x, y, xerr,yerr,fixedHyperparams, weight=None ):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.yerr = yerr
        self.fixedHyperparams=fixedHyperparams
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)

    def __call__(self, A, mass, tau):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixedHyperparams.values() #best_fit_gp
        bkgFunc=model_gp((p0,p1,p2), self.x, self.xerr)
        logL=0

        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            binWeight = self.weight[ibin]
            logL += -simpleLogPoisson(data/binWeight, bkg/binWeight)
        try:
            logL
            return logL
        except:
            return np.inf


def fit_gp_sigRecon(lnprob,mass, trial=50, Print = True):
    bestval = np.inf
    bestargs = (0, 0, 0)
    passedFit = False
    numRetries = 0
    for i in range(trial):
        init0 = np.random.random() * 1e11
        init1 = np.random.random() * mass*(1+massReconstructionWindow)
        init2 = np.random.random() * 120
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0, errordef = 0.5,
                  A = init0, mass = init1, tau = init2,
                  error_A = 1., error_mass = 1., error_tau = 1.,
                  limit_A = (1e4, 1e11), limit_mass = (mass*(massReconstructionWindow-1), mass*(massReconstructionWindow+1)), limit_tau = (30, 120))
        fit = m.migrad()
        if m.fval < bestval:
            bestval = m.fval
            bestargs = m.args
            print (bestargs)

    if Print:
        print ("min LL", bestval)
        print ("best fit vals",bestargs)
    return bestval, bestargs

#
#-------UA2
def model_UA2(t, params, xErr):
    p0, p1, p2, p3 = params
    sqrts = 13000.
    return p0 / (t/sqrts)**p1 * np.exp(-p2* (t/sqrts)-p3 *(t/sqrts)**2) *xErr

class logLike_UA2:
    def __init__(self, x, y, xe, weight=None):
        self.x = x
        self.y = y
        self.xe = xe
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)

    def __call__(self, p0, p1, p2, p3):
        params = p0, p1, p2, p3
        bkgFunc = model_UA2(self.x, params, self.xe)
        logL = 0

        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = bkgFunc[ibin]
            binWeight = self.weight[ibin]
            logL += -simpleLogPoisson(data/binWeight, bkg/binWeight)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_UA2(num,lnprob,initParam=None,initFitParam=None, initRange=None,  Print=True):
#initialization
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)
    if initParam==None: # use default values if it's not specifiied
        # no tag
        #initParam=[9.6, -1.67, 56.87,-75.877]
        print("Using default initParam")
        #------btagged 1
        initParam=[0.5145306090125814, -2.1711845084356156, 37.26540673784112, -66.86178995858384]
    if initFitParam==None:
        initFitParam=[100, 10, 100, 300]
        #------btagged 2
        #initFitParam=[1000, 10, 100, 200]
        #initParam=[0.2230885992853473, -1.6173406469678184, 44.10320875775574, -129.8708750637629]

        #-----trijet
        #initFitParam=[10000000, 50, 200, 500]
         #non use Scaled
        #initParam=(297194.05479060113, 0.33939799509016044, 96.16372727354921, 1.9280619077735317)
        #useScaled
        #initParam=(923489.4363469365, 0.07338929885136791, 102.61856227867057, 3.338702006368571)
        #initParam=(923492.5666110474, 0.07340343406897532, 102.61856087361235, 3.3674253868660884)
        #initParam=(2114090.3221715163, -0.10249143346059597, 110.96753410590006, -41.90626047681503)

    if initRange==None:
        initRange=[(2000000, 50000000.),(-10, 10),(-100, 600.),(-300, 300.)]

    print("UA fitting using using inital params: ", initParam)
    print("UA fitting using param range: ", initRange)


    for i in range(num):
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = initParam[0], p1 = initParam[1], p2 = initParam[2], p3= initParam[3],
                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1, error_p3 = 1e-1,
                  limit_p0 = initRange[0], limit_p1 = initRange[1], limit_p2 = initRange[2], limit_p3=initRange[3])
        m.migrad()
        print("trial ", i, " fit params: ", m.args)
        print("-log likelihood: ", m.fval)
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args
        initParam=[x * np.random.random() for x in initFitParam]

    if best_fit_params == (0., 0., 0.):
        print("------------------fit failed ----------------")

    if Print:
        print("fit function UA2")
        print ("min LL", prettyfloat(minLLH))
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
            bkgFunc[ibin]
            logL += -simpleLogPoisson(data, bkgFunc)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_4ff(num, lnprob, initParam=None, initFitParam=None, initRange=None,Print=True): #use this to minimize for the best fit function
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)

    if initParam==None: # use default values if it's not specifiied
        initParam=[52, 53, -1.199, 1]
    for i in range(len(initParam)):
        initParam[i] = initParam[i] *np.random.random()
    if initRange==None:
        initRange=[(-10, 100000.),(-100., 100.),(-100., 100.),(-100., 100.)]

    print("UA fitting using using inital params: ", initParam)
    print("UA fitting using param range: ", initRange)
    for i in range(num):
        m = Minuit(lnprob, throw_nan=False, pedantic=False, print_level=0,
                p0=initParam[0], p1=initParam[1], p2=initParam[2], p3=initParam[3],
                   error_p0=1e-2, error_p1=1e-1, error_p2=1e-1, error_p3=1e-1,
                   limit_p0=(0, 100.), limit_p1=(-100., 100.), limit_p2=(-100., 100.),limit_p3=(-100.,100.))
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args
    if Print:
        print("fit function 4ff")
        print ("min LL", minLLH)
        print ("best fit vals", best_fit_params)
    return minLLH, best_fit_params

class logLike_gp_sig_fixedH:
    def __init__(self, x, y, xerr, fixedHyperparams):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.fixedHyperparams=fixedHyperparams
    def __call__(self, Num, mu, sigma):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixedHyperparams.values()
        kernel = Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            signal = sig_model(self.x, Num, mu, sigma, self.xerr)
            return -gp.lnlikelihood(self.y - model_gp((p0,p1,p2), self.x, self.xerr)-signal)
        except:
            return np.inf

def fit_gp_sig_fixedH_minuit(lnprob, mass, trial, Print = True):
    print("reconstruct mass: ", mass)
    bestval = np.inf
    bestargs = (0, 0, 0)
    passedFit = False
    numRetries = 0
    for i in range(trial):
        init0 = np.random.random() * 500000000.
        init1 = np.random.random() * mass*(1+massReconstructionWindow)
        init2 = np.random.random() * 200.
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0, errordef = 0.5,
                  Num = init0, mu = init1, sigma = init2,
                  error_Num = 1., error_mu = 1., error_sigma = 1.,
                  limit_Num = (1, 500000000000), limit_mu = (mass*(1-massReconstructionWindow), mass*(1+massReconstructionWindow)), limit_sigma = (20, 700))
        fit = m.migrad()
        if m.fval < bestval:
            bestval = m.fval
            bestargs = m.args

    if Print:
        print ("min LL", bestval)
        print ("best fit vals",bestargs)
    return bestval, bestargs


#def sig_model(x, N=1e5, mass=2000., width=100., xErr=0.1):
#    return N*(np.exp(-(x-mass)**2/2/width/width)/np.sqrt(2*np.pi)/width)*xErr
def customSignalModel(N, yTemp):
    #probably don't need x?
    #N=40000
    return N* yTemp

class logLike_gp_customSigTemplate:
    def __init__(self, x, y, xerr, xTemplate, yTempNorm, fixedHyper):
        self.x = x
        self.y = y
        self.xerr = xerr
        #self.hist=hist
        self.xTemplate=xTemplate
        self.yTempNorm= yTempNorm
        self.fixedHyper=fixedHyper

    def __call__(self, Num):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixedHyper.values()
        kernel = Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            signal = customSignalModel(Num, self.yTempNorm)
            #perhaps the lnlikelihood becasme -ve here and is undefined
            return -gp.lnlikelihood(self.y - model_gp((p0,p1,p2), self.x, self.xerr)-signal)
        except:
            return np.inf

class logLike_gp_customSigTemplate_diffLog:
    def __init__(self, x, y, xerr, xTemplate, yTempNorm, fixedHyper,weight=None, bkg=None):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.xTemplate=xTemplate
        self.yTempNorm= yTempNorm
        self.fixedHyper=fixedHyper
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)
        if bkg is not None:
            self.bkgFunc=bkg
        else:
            self.bkgFunc=model_gp((p0,p1,p2), self.x, self.xerr)

    def __call__(self, Num):
        Amp, decay, length, power, sub, p0, p1, p2 = self.fixedHyper.values() #best_fit_gp
        signal = customSignalModel(Num, self.yTempNorm)
        print("Num: ", Num)
        print("signal:", signal)
        logL=0
        tetsum=0
        for ibin in range(len(self.y)):
            data = self.y[ibin]
            bkg = self.bkgFunc[ibin]
            sig = signal[ibin]
            binWeight = self.weight[ibin]
            testSum=data/binWeight- (bkg+sig)/binWeight
            logL += -simpleLogPoisson(data/binWeight, (bkg+sig)/binWeight)
        print("logL: ", logL)
        print("testSum: ", testSum)
        try:
            logL
            return logL
        except:
            return np.inf

def fit_gp_customSig_fixedH_minuit(lnprob, trial, Print = True):
    bestval = np.inf
    bestargs = (0, 0, 0, 0)
    passedFit = False
    numRetries = 0
    for i in range(trial):
        init0 = np.random.random() * 5000000000.
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0, errordef = 0.5,
                  Num = init0,
                  error_Num = 1.,
                  limit_Num = (400, 5000000000))
        m.migrad()
        if m.fval < bestval and m.fval!=0.0:
            bestval = m.fval
            bestargs = m.args
        print("log likelihood: ", m.fval)
        print("custom fit params:", m.args)

    if Print:
        print ("min LL", bestval)
        print ("best fit vals custom: ",bestargs)
    return bestval, bestargs



def y_bestFit3Params(x, y, xerr, likelihoodTrial):
    lnProb = logLike_3ff(x,y,xerr)
    minimumLLH, best_fit_params = fit_3ff(100, lnProb) #finding the best fit param by the minimum likelihood
    fit_mean = model_3param(best_fit_params, x, xerr) # fitting using the best fit params
    return fit_mean

def y_bestFitGP(x,xPred,y,xerr,yerr, likelihoodTrial, kernelType="bkg",mass=None, bkgDataParams=None):
    if kernelType=="bkg":
        lnProb = logLike_minuit(x, y, xerr)
        min_likelihood, best_fit = fit_gp_minuit(likelihoodTrial, lnProb)
        #print(best_fit)
        if bkgDataParams:
            # only use the fit function results but not the kernel parameter results from fit_gp_minuit
            fit_pars=[bkgDataParams[x] for x in FIT3_PARS]
            kargs = {x:y for x, y in bkgDataParams.items() if x not in FIT3_PARS}
            best_fit = {x:y for x, y in bkgDataParams.items() if x not in FIT3_PARS}
            # changing the bkgnd kernel hyper param value of the best fit
        else:
            fit_pars = [best_fit[x] for x in FIT3_PARS]
            kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
    #bkgnd kernel
        kernel = get_kernel(**kargs)
#
    elif kernelType =="sig":
        #lnProb = logLike_minuit_sig(x, y, xerr)
        lnProb =logLike_minuit_sigWithWindow(x, y, xerr)
        #lnProb = logLike_gp_fitgpsig(x, y, xerr)
        #lnProb = logLike_minuit(x, y, xerr)
        if bkgDataParams:
            fit_pars=[bkgDataParams[x] for x in FIT3_PARS]
            kargs = {x:y for x, y in bkgDataParams.items() if x not in FIT3_PARS}
            best_fit = {x:y for x, y in bkgDataParams.items() if x not in FIT3_PARS}

#-----specific setup for example mR500
            #min_likelihood, best_fit = fit_gpSig_minuit(likelihoodTrial, lnProb, (9424207558.974304, 23.36140499375173, 212320.71588945185, 0.934086263121374, 677.3782035685553,5.835978239102488, 19.11598702209463, -2.3008859313936085))
            min_likelihood, best_fit = fit_gpSig_minuit(likelihoodTrial, lnProb, mass, list(bkgDataParams.values()))
            print ("min likelihood:", min_likelihood)
            print(best_fit)
            fit_pars = [best_fit[x] for x in FIT3_PARS]
            kargs = {x:y for x, y in best_fit.items() if x not in FIT3_PARS}
        #signal kernel
            kernel = get_kernel_Xtreme(**kargs)
            #kernel = get_kernel(**kargs)

    #if kernelType =="bkg":
    gp = george.GP(kernel, mean=Mean(fit_pars), fit_mean = True)
    gp.compute(x, yerr)
    mu, cov = gp.predict(y, xPred)
    return mu, cov , best_fit
    #return 0.0, 0.0
def psuedoTest(pseudoTestNum, yBkgFit, yErrBkgFit,fit_mean, yBkg, yErrBkg,mu_xBkg):
    chi2FitList=[]
    chi2GPList=[]

    beforeWeightY = yErrBkg**2
    weightY = yBkgFit/beforeWeightY
    beforeWeightYFit=yErrBkgFit**2
    weightFitY = yBkgFit/beforeWeightYFit

    for i in range(pseudoTestNum):
        #toyList_fit_mean=np.random.poisson(fit_mean)
        #toyList_muGP=np.random.poisson(mu_xBkg)

        toyList_fit_mean_bw= np.random.poisson(fit_mean/weightY)
        toyList_muGP_bw=np.random.poisson(mu_xBkg/weightFitY)

#finding the chi2 for the list of toy thrown
        fitSignificance, chi2fit=res_significance(beforeWeightY,toyList_fit_mean_bw )
        chi2FitList.append(chi2fit)
        GPSignificance, chi2GP=res_significance(beforeWeightYFit, toyList_muGP_bw)
        chi2GPList.append(chi2GP)
    chi2FitList=np.array(chi2FitList)
    chi2GPList=np.array(chi2GPList)
    return  chi2FitList, chi2GPList


##def runGP_SplusB(ys, xs_train, xerr_train, xs_fit, xerr_fit, hyperParams):
##    Amp, decay,  lengthscale, power, sub, p0, p1, p2, A, mass, tau = hyperParams
##    kernel_sig = get_kernel_SigOnly(A, mass, tau)
##    kernel_bkg = get_kernel(Amp, decay, lengthscale, power, sub)
##
##
##    kernel = kernel_bkg + kernel_sig
##    gp = george.GP(kernel)
##    gp.compute(xs_train, np.sqrt(ys))
##
##    MAP_GP, cov_GP = gp.predict( ys - model_3param(xs_train, (p0, p1, p2), xerr_train), xs_fit)
##    MAP_GP = MAP_GP + model_3param(xs_fit,(p0, p1, p2), xerr_fit)
##
##    K1 = kernel_bkg.get_value(np.atleast_2d(xs_train).T)
##    MAP_sig = np.dot(K1, gp.solver.apply_inverse(ys - model_3param(xs_train,(p0, p1, p2), xerr_train))) + model_3param( xs_train, (p0, p1, p2), xerr_train)
##    K2 = kernel_sig.get_value(np.atleast_2d(xs_train).T)
##    MAP_bkg = np.dot(K2, gp.solver.apply_inverse(ys-model_3param( xs_train, (p0, p1, p2),xerr_train)))
##
##    return MAP_GP, MAP_sig, MAP_bkg

#original
#def runGP_SplusB(ys, xs_train, xerr_train, xs_fit, xerr_fit, hyperParams):
#    Amp, decay,  lengthscale, power, sub, p0, p1, p2, A, mass, tau = hyperParams
#
#    kernel_sig = get_kernel_SigOnly(A, mass, tau)
#    kernel_bkg = get_kernel(Amp, decay, lengthscale, power, sub)
#    kernel = kernel_bkg + kernel_sig
#    gp = george.GP(kernel)
#    gp.compute(xs_train, np.sqrt(ys))
#
#    MAP_GP, cov_GP = gp.predict( ys - model_3param(xs_train, (p0, p1, p2), xerr_train), xs_fit)
#    MAP_GP = MAP_GP + model_3param(xs_fit,(p0,p1,p2),xerr_fit)
#
#    K1 = kernel_bkg.get_value(np.atleast_2d(xs_train).T)
#    MAP_sig = np.dot(K1, gp.solver.apply_inverse(ys - model_3param(xs_train,(p0,p1,p2),xerr_train))) + model_3param(xs_train,(p0,p1, p2), xerr_train)
#    K2 = kernel_sig.get_value(np.atleast_2d(xs_train).T)
#    MAP_bkg = np.dot(K2, gp.solver.apply_inverse(ys-model_3param(xs_train, (p0,p1, p2), xerr_train)))
#
#    return MAP_GP, MAP_sig, MAP_bkg
##why is this weird
def runGP_SplusB(ys, xs_train, xerr_train, xs_fit, xerr_fit, hyperParams):
    Amp, decay,  lengthscale, power, sub, p0, p1, p2, A, mass, tau,l = hyperParams

    kernel_sig = get_kernel_SigOnly(A, mass, tau,l)
    kernel_bkg = get_kernel(Amp, decay, lengthscale, power, sub)
    kernel = kernel_bkg + kernel_sig
    gp = george.GP(kernel)
    gp.compute(xs_train, np.sqrt(ys))

    MAP_GP, cov_GP = gp.predict( ys - model_3param((p0, p1, p2), xs_train, xerr_train), xs_fit)
    MAP_GP = MAP_GP + model_3param((p0,p1,p2),xs_fit,xerr_fit)
#K1 is : covariance matrix
    K1 = kernel_bkg.get_value(np.atleast_2d(xs_train).T)

#MAP_sig
    MAP_sig = np.dot(K1, gp.solver.apply_inverse(ys - model_3param((p0,p1,p2),xs_train, xerr_train))) + model_3param((p0,p1, p2),xs_train, xerr_train)
    K2 = kernel_sig.get_value(np.atleast_2d(xs_train).T)
    MAP_bkg = np.dot(K2, gp.solver.apply_inverse(ys-model_3param((p0,p1, p2), xs_train,xerr_train)))

    return MAP_GP, MAP_sig, MAP_bkg
