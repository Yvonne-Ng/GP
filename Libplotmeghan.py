#!/usr/bin/env python3

'''
this defines all the functions in python which are called later by the run functions
'''
from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp,LocalGaussianKernel#, ExpSquared#, ExpSquaredCenteredKernel, ExpSquaredKernel
from iminuit import Minuit
import scipy.special as ssp
import inspect
import math

import random

from numpy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

FIT3_PARS = ['p0','p1','p2']
FIT4_PARS = ['p0', 'p1', 'p2','p3']

#General Function
#def gauss(x, *p):
#    a, b, c, d = p
#    y = a*np.exp(-np.power((x - b), 2.)/(2. * c**2.)) + d

  #  return y

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
        print("print x in getDataPoints:", x)
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
        chi2 += ((nD - nB) ** 2 / abs(nB)) 
    #print("length of Zval:", len(zvals))
    zvals=np.array(zvals)

    return zvals, chi2
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
        # steps = np.append(np.diff(t), np.diff(t)[-1])
        # print(steps)
        #print("size of t",t.shape)
        #print("size of xErr", xErr.shape)
        vals = (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))
        #print(vals)
        return vals

def get_kernel(Amp, decay, length, power, sub): # getting the typical background kernel
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)
def get_kernel_SigOnly(A, mass, tau):
    return A* LocalGaussianKernel(mass, tau)

def get_kernel_Xtreme(Amp, decay, length, power, sub, A, mass, tau): # getting the signal+bkgnd kernel
    #kernelsig = Amp * ExpSquaredCenteredKernel(m=mass, t=tau)
    kernelsig= A* LocalGaussianKernel(mass, tau)
    kernelbkg = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
    return kernelsig+kernelbkg
    #return kernelbkg
def get_xy_pts(group):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values'])
    edges = np.asarray(group['edges'])
    errors = np.asarray(group['errors'])
    center = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    print("centers in function: ", center)
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
        ipower = np.random.random() * 1.0
        isub = np.random.random() * 1.0
        ip0 = guesses['p0']
        ip1 = guesses['p1']
        ip2 = guesses['p2']
        #print(lnprob)
        #print(inspect.getargspec(logLike_minuit))

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
                   limit_length = (1, 1e8),
                   limit_power = (0.0001, 100),
                   limit_sub = (0.001, 1e6),
                   #limit_p0 = bound('p0', neg=false),
                   #limit_p1 = bound('p1', neg=true),
                   #limit_p2 = bound('p2', neg=true))
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
def fit_gpSig_minuit(num, lnprob, initParams):

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
        iA = (2500000 *np.random.random())
        itau = (50 * np.random.random())
        iMass = (500 * np.random.random())

        m = Minuit(lnprob, throw_nan = True, pedantic = True,
                   print_level = 0, errordef = 0.5,
                   Amp = iamp,decay = idecay, length = ilength,power = ipower,sub = isub,
                   p0 = ip0, p1 = ip1, p2 = ip2,
                   A=iA, tau=itau, mass=iMass,
                   error_Amp = 1e1,
                   error_decay = 1e1,
                   error_length = 1e1,
                   error_power = 1e1,
                   error_sub = 1e1,
                   error_p0 = 1e-2, error_p1 = 1e-2, error_p2 = 1e-2,
                   error_A = 1e-2, error_tau = 1e-2, error_mass = 1e-2,
                   limit_Amp = (initParams[0], initParams[0]*1.001),#bound('amp'),
                   limit_decay = (initParams[1], initParams[1]*1.001),
                   limit_length = (initParams[2], initParams[2]*1.001),
                   limit_power = (initParams[3], initParams[3]*1.001),
                   limit_sub = (initParams[4], initParams[4]*1.001),
                   #limit_decay = (0.01, 1000),
                   #limit_length = (1, 1e8),
                   #limit_power = (0.0001, 100),
                   #limit_sub = (0.001, 1e6),
                   #limit_p0 = bound('p0', neg=false),
                   #limit_p1 = bound('p1', neg=true),
                   #limit_p2 = bound('p2', neg=true))
                   #limit_p0 = (-100,1000000),
                   #limit_p1 = (-100,100),
                   #limit_p2 = (-100,100),
                   limit_p0 = (initParams[5],initParams[5]*1.001),
                   limit_p1 = (initParams[6], initParams[6]*1.001),
                   limit_p2 = (initParams[7],initParams[7]*0.9999),
                   limit_A = (500, 2500000),
                   limit_tau = (20, 300),
                   limit_mass=(400, 550))
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
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr

    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2, A=0, mass=0, tau=0):
        #Amp, decay, length, power, sub, p0, p1, p2 = fixedHyperparams
        kernel1 = Amp * MyDijetKernelSimp(a=decay, b=length, c=power, d=sub)
        kernel2 = A * LocalGaussianKernel(mass, tau)
        kernel = kernel1 #+ kernel2
        #kernel=kernel1
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return np.inf

def fit_gp_fitgpsig_minuit(num, lnprob, initParams,Print=True):
    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0,0,0,0)
    for i in range(num):
        #iAmp= np.random.random() *5701461179.0
        #iDecay = np.random.random() *0.64
        #ilength = np.random.random() *5e5
        #ipower =np.random.random() * 1.0
        #isub= np.random.random() * 1.0
        ##ip0=np.random.random() * 0.23
        ##ip1= np.random.random() *0.46
        ##ip2=np.random.random() *0.89
        #ip0=np.random.random() * 0.23
        #ip1= np.random.random() *28.38
        #ip2=np.random.random() *-2.538
        #iA = 0
        #imass=np.random.random() *500
        #itau= np.random.random() *100000000
#signal 
        #iAmp= np.random.random() *2880395166.7137227
        #iDecay =np.random.random() * 24.80421529029139
        #ilength = np.random.random() *1.0000000666133808
        #ipower = np.random.random() *0.001
        #isub= np.random.random() *632.2987
        #ip0=np.random.random() *7.099
        #ip1= np.random.random() *19.85
        #ip2=np.random.random() *-2.24
        #iA = (np.random.random() *2)*0.000001
        #imass=np.random.random() *np.random.random() *500
        #itau= np.random.random() *np.random.random() *100000000

        #iAmp=np.random.random() *2880395166.7137227 
        #iDecay =np.random.random() *24.80
        #ilength =np.random.random() * 1.00
        #ipower =np.random.random() * 0.0010
        #isub= np.random.random() *632.298
        #ip0=np.random.random() *7.09
        #ip1=np.random.random() * 19.83
        #ip2=np.random.random() *-2.24
        #iA =np.random.random() *0#(np.random.random() *2)*0.000001
        #imass=np.random.random() *500
        #itau= np.random.random() *100000000

        iAmp=np.random.random() * initParams[0]
        iDecay =np.random.random() * initParams[1]
        ilength =np.random.random() *initParams[2]
        ipower =np.random.random() * initParams[3]
        isub= np.random.random() *initParams[4]
        ip0=np.random.random() *initParams[5]
        ip1=np.random.random() * initParams[6]
        ip2=np.random.random() * initParams[7]
        iA =np.random.random() *0#(np.random.random() *2)*0.000001
        imass=np.random.random() *500
        itau= np.random.random() *100000000

        m = Minuit(lnprob, throw_nan=True, pedantic=True, print_level=0, errordef=0.5, Amp=iAmp, decay=iDecay, length=ilength, power=ipower, sub=isub, p0=ip0, p1=ip1, p2=ip2, A=iA, mass=imass, tau=itau, error_Amp=1e1, error_decay=1e1, error_length=1e1, error_power=1e1, error_sub=1e1, error_p0=1e-2, error_p1=1e-2, error_p2=1e-2, error_A=1., error_mass=1., error_tau=1., limit_Amp=(-100000,100000000000), limit_decay=(0.01, 1000), limit_length=(1, 1e8), limit_power=(0.001, 1000), limit_sub=(0.001, 1e6), limit_p0=(-100,1000000.), limit_p1=(-100., 100.), limit_p2=(-100., 100), limit_A=(-0.1,0.1), limit_mass=(450, 550), limit_tau=(-500, 500))

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

def fit_3ff(num,lnprob, Print=True):
    minLLH = np.inf
    best_fit_params = (0., 0., 0.)
    for i in range(num):
        #init0 =  0.2336
        #init1 = np.random.random() * 0.46
        #init2 = np.random.random() * 0.8901

        init0 =  np.random.random() *0.069
        init1 = np.random.random() * 22.43
        init2 = np.random.random() * -2.807

        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = init0, p1 = init1, p2 = init2,
                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1, 
                  limit_p0 = (0, 100.), limit_p1 = (-100., 100.), limit_p2 = (-100., 100.))
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args 
    if Print:
        print("fit function")
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
       # init0 = 52.
       # init1 = 53.
       # init2 = -1.199
       # init3 = 1
        init0 = 9.6
        init1 = -1.67
        init2 = 56.87
        init3 = -75.877
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0,
                  p0 = init0, p1 = init1, p2 = init2, p3= init3,
                  error_p0 = 1e-2, error_p1 = 1e-1, error_p2 = 1e-1, error_p3 = 1e-1,
                  limit_p0 = (-100000, 1000000.), limit_p1 = (-100., 100.), limit_p2 = (-100., 100.), limit_p3=(-100.,100.))
        m.migrad()
        if m.fval < minLLH:
            minLLH = m.fval
            best_fit_params = m.args
    if Print:
        print("fit function UA2")
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
            bkgkgFunc[ibin]
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
                print("fit function 4ff")
                print ("min LL", minLLH)
                print ("best fit vals", best_fit_params)
            return minLLH, best_fit_params
#-------GP
#----------------------Hiding the procesing of the best fit function values#------

def y_bestFit3Params(x, y, xerr, likelihoodTrial):
    lnProb = logLike_3ff(x,y,xerr)
    minimumLLH, best_fit_params = fit_3ff(100, lnProb) #finding the best fit param by the minimum likelihood 
    fit_mean = model_3param(x, best_fit_params, xerr) # fitting using the best fit params 
    return fit_mean

def y_bestFitGP(x,y,xerr,yerr, likelihoodTrial, kernelType="bkg", bkgDataParams=None):
    if kernelType=="bkg":
        lnProb = logLike_minuit(x, y, xerr)
        #print ("in run bkg lnprob:", lnProb)
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
        print(kernel.get_parameter_names())
#
    elif kernelType =="sig":
        lnProb = logLike_minuit_sig(x, y, xerr)
        #lnProb = logLike_gp_fitgpsig(x, y, xerr)
        #lnProb = logLike_minuit(x, y, xerr)


        #print ("in run sig lnprob:", lnProb)
        #print("signal lnprob: ", lnProb(9424207558.974304, 23.36140499375173, 212320.71588945185, 0.934086263121374, 677.3782035685553,5.835978239102488, 19.11598702209463, -2.3008859313936085))

        min_likelihood, best_fit = fit_gpSig_minuit(likelihoodTrial, lnProb, (9424207558.974304, 23.36140499375173, 212320.71588945185, 0.934086263121374, 677.3782035685553,5.835978239102488, 19.11598702209463, -2.3008859313936085))
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
    mu, cov = gp.predict(y, x)
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
    Amp, decay,  lengthscale, power, sub, p0, p1, p2, A, mass, tau = hyperParams

    kernel_sig = get_kernel_SigOnly(A, mass, tau)
    kernel_bkg = get_kernel(Amp, decay, lengthscale, power, sub)
    kernel = kernel_bkg + kernel_sig
    gp = george.GP(kernel)
    gp.compute(xs_train, np.sqrt(ys))

    MAP_GP, cov_GP = gp.predict( ys - model_3param(xs_train, (p0, p1, p2), xerr_train), xs_fit)
    MAP_GP = MAP_GP + model_3param(xs_fit,(p0,p1,p2),xerr_fit)
#K1 is : covariance matrix
    K1 = kernel_bkg.get_value(np.atleast_2d(xs_train).T)

#MAP_sig
    MAP_sig = np.dot(K1, gp.solver.apply_inverse(ys - model_3param(xs_train,(p0,p1,p2),xerr_train))) + model_3param(xs_train,(p0,p1, p2), xerr_train)
    K2 = kernel_sig.get_value(np.atleast_2d(xs_train).T)
    MAP_bkg = np.dot(K2, gp.solver.apply_inverse(ys-model_3param(xs_train, (p0,p1, p2), xerr_train)))
    
    return MAP_GP, MAP_sig, MAP_bkg
