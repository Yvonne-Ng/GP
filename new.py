
def customSignalModel(N, hist=None):
    #probably don't need x? 
    return hist.Scale(N)

class logLike_gp_customSigTemplate:
    def __init__(self, x, y, xerr, hist):
        self.x = x
        self.y = y
        self.xerr = xerr
        self.hist=hist
    def __call__(self, N):
        Amp, decay, length, power, sub, p0, p1, p2 = fixedHyperparams.values()
        kernel = Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)
        gp = george.GP(kernel)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            signal = customSignalModel(Num,self.hist)
            return -gp.lnlikelihood(self.y - model_gp((p0,p1,p2), self.x, self.xerr)-signal)
        except:
            return np.inf        

    
def fit_gp_customSig_fixedH_minuit(lnprob, Print = True):
    bestval = np.inf
    bestargs = (0, 0, 0)
    passedFit = False
    numRetries = 0
    for i in range(1):
        init0 = np.random.random() * 5000.
        m = Minuit(lnprob, throw_nan = False, pedantic = False, print_level = 0, errordef = 0.5,
                  Num = init0,  
                  error_Num = 1.,
                  limit_Num = (1, 50000000)) 
        fit = m.migrad()
        if m.fval < bestval:
            bestval = m.fval
            bestargs = m.args     

    if Print:
        print ("min LL", bestval)
        print ("best fit vals",bestargs)
    return bestval, bestargs
