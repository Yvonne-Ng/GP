#!/usr/bin/env python3

'''
this defines all the functions in python which are called later by the run functions
'''
from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp#, ExpSquaredCenteredKernel, ExpSquaredKernel
from iminuit import Minuit
import scipy.special as ssp
import inspect

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
        #print("size of t",t.shape)
        #print("size of xErr", xErr.shape)
        vals = (p0 * ((1.-t/sqrts)**p1) * (t/sqrts)**(p2))
        #print(vals)
        return vals

def get_kernel(Amp, decay, length, power, sub):
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)

def get_kernel_Xtreme(Amp, decay, length, power, sub, mass, tau):
    kernelsig = Amp * ExpSquaredCenteredKernel(m=mass, t=tau)
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

def fit_gp_fitgpsig_minuit(lnprob, Print=True):
    bestval = np.inf
    bestargs = (0, 0, 0)
    passedFit = False
    numRetries = 0

    for i in range(100):
        iAmp= 5701461179.0
        iDecay = 4000.
        ilength = 200.
        ipower = 1.0
        isub= 1.0
        ip0= 0.23
        ip1= 0.46
        ip2= 0.89
        iA = 5.0
        imass=500
        itau= 200

        print(lnprob)
        print(inspect.getargspec(logLike_gp_fitgpsig))

        m = Minuit(lnprob, throw_nan=False, pedantic=True, print_level=0, #errordef=0.5,
                   Amp=iAmp, decay=iDecay, length=ilength, power=ipower, sub=isub, p0=ip0, p1=ip1, p2=ip2, A=iA, mass=imass, tau=itau,
                   error_Amp=1e1, error_decay=1e1, error_length=1e1, error_power=1e1, error_sub=1e1, error_p0=1e-2, error_p1=1e-2, error_p2=1e-2, error_A=1., error_mass=1., error_tau=1.,
                   limit_Amp=(-100000,100000), limit_decay=(0.01, 1000), limit_length=(100, 1e8), limit_power=(0.01, 1000), limit_sub=(0.01, 1e6), limit_p0=(0,100.), limit_p1=(-100., 100.), limit_p2=(-100., 100), limit_A=(0,100), limit_mass=(1000, 7000), limit_tau=(100, 500))

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
        init0 =  0.2336
        init1 = np.random.random() * 0.46
        init2 = np.random.random() * 0.8901
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
                print ("min LL", minLLH)
                print ("best fit vals", best_fit_params)
            return minLLH, best_fit_params
#-------GP

