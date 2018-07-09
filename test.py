
class logLike_UA2:
    def __init__(self, x, y, xe, weight=None):
        self.x = x
        self.y = y
        self.xe = xe
        if weight is not None:
            self.weight=weight
        else:
            self.weight=np.ones(self.x.size)
        #print("self.weight", self.weight)
        
    def __call__(self, p0, p1, p2, p3):
        params = p0, p1, p2, p3
        bkgFunc = model_UA2(self.x, params, self.xe)
        logL = 0
        #print ("dataY: ", self.y)
        #print ("bkgFunc: ", bkgFunc)
        #print("dataY/weight: ",self.y/self.weight)
        #print("bkgFunc/weight: ",bkgFunc/self.weight)
        
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
