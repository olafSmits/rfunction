# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 17:15:08 2013
"""

from __future__ import division
import numpy as np
import sympy.mpmath as mp
import Rfunc_series as Rfunc
import matplotlib.pylab as plt
from numpy import newaxis, vectorize, power, arange
from sympy.mpmath import mpf

mp.mp.pretty = True
mp.mp.dps = 20

fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)




class Rfunc_CNCT(Rfunc.Rfunc_series):
    maxA = 10
    maxK = 9
    def setParameter(self, nterms = None, maxA = None, maxK = None):
        if maxA is not None:        
            self.maxA = maxA
        if maxK is not None:
            self.maxK = maxK
        if nterms is not None:
            self.nterms = nterms
    def genWijngaardenTerms(self):  
        if 2**2*(self.maxA+1) - 1 >= self.nterms:
            print """
Warning: self.maxA is relatively low. Lack of accuracy may occur.
Advice: increase nterms or lower maxA"""
                    
        self.wijngaardenArray = \
            np.fromfunction(lambda  j, k: 2**k*(j+1) -1,(self.maxA, self.maxK)
                            ,dtype = np.int64)
        self.wijngaardenArray[self.wijngaardenArray >= self.nterms] = -1
        self.wijnTerms = np.unique(self.wijngaardenArray)
        if -1 in self.wijnTerms:
            self.wijnTerms = np.delete(self.wijnTerms,0)
    def extractWijngaardenFromLDA(self):
        if not hasattr(self, 'lda') or len(self.lda) != self.nterms:
            self.genLDA()
        def _f(i, j):
            ldaDICT = dict((k, self.lda[k][j])  for k in self.wijnTerms)
            ldaDICT[-1] = mpf(0)
            def g(n): 
                return ldaDICT[n]
            _g = np.vectorize(g)
            return _g(i)
        ldaTemp = np.ones((self.maxA, self.maxK, self.lda.shape[1]), 
                          dtype = object)
        for i in range(self.lda.shape[1]):
            ldaTemp[:,:,i] = _f(self.wijngaardenArray, i)
        self.lda = ldaTemp
        wijngaardenFactor = np.power(-1,np.arange(0, self.maxA))[:,newaxis]* \
                                (2**np.arange(0, self.maxK))
        self.lda = self.lda*wijngaardenFactor[:,:,newaxis]
        
    def genGamma(self):
        if not hasattr(self, 'wijnTerms') \
            or self.wijngaardenArray.shape != (self.maxA, self.maxK):
            self.genWijngaardenTerms()
        gDict = dict((k, fgamma(self.gtot + mpf(str(k)))/fgamma(self.gtot)) \
                        for k in self.wijnTerms)
        def f(i, j):
            gamDict = self.__genGammaDict(j, gDict)
            gamDict[-1] = mpf('0')
            def g(n):
                return gamDict[n]
            _g = vectorize(g)
            return _g(i)

        self.gamma = fmfy(np.ones((self.maxA, self.maxK, self.scaledVolt.size)))
        for j in xrange(self.scaledVolt.size):
            self.gamma[:,:,j] = f(self.wijngaardenArray, j)
    def __genGammaDict(self, j, gDict):
        if self.isZeroT:
            return dict((k, self.scaledVolt[j]**k /gDict[k]) for k in self.wijnTerms)
        else:
            fg = fgamma(self.scaledVolt[j])
            return dict((k, fgamma(self.scaledVolt[j] + mpf(str(k))) / \
                                (fg*gDict[k])) for k in self.wijnTerms)



            
    def mergeLDAandGamma(self):
        self.extractWijngaardenFromLDA()
        self.lauricella_terms = np.sum(self.lda[:,:,newaxis,:]*
                                        self.gamma[...,newaxis], axis = 1)
                                        
        self.lauricella_terms2 = self.lauricella_terms.copy()
        self.lauricella = levin_acceleration(self.lauricella_terms)

def levin_acceleration(L, beta = 1.):
    def LB(n, k):
        return (beta + n + k)*power(beta + n + k, k-1)/power(beta+n+k+1,k)
    def SB(n, k):
        return (beta+n+k)*(beta+n+k-1)/((beta+n+2*k)*beta+n+2*k-1)
    
    ### remainder estimator
    rem = 1/L[1:]
    #rem = L[:-1]
    #rem = L[1:]*L[:-1]  (L[1:] - L[:-1])
    ###
    denominator = recursive_generator(rem, SB)
    rem = 1/L[1:]
    numerator = recursive_generator(L.cumsum(axis=0)[:-1]*rem, SB)
    return (numerator / denominator)
    
def recursive_generator(L, f):
    M = L.shape[0]
    for i in range(1,M-1):
        L[:-i] = L[1:M-i+1] - f(fmfy(arange(0,M-i)), i)[:,newaxis,newaxis]* L[:-i]
    return L[0,:,:]
    
def log_2(n):
    return (-1)**n / n    
    
if __name__ == '__main__':
    import InputParameters as BP
    VOLTRANGE = fmfy(np.linspace(0.01,50,5)) * BP.GLOBAL_VOLT
    basedist = mpf(1.0)/mpf(10**6)
    distance = np.linspace(.5, 1.0, 3) * basedist
    distance2 = np.ones_like(distance) * basedist
    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
              "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
                 "x":[distance2, -distance, distance2, -distance]}
    A = BP.base_parameters(example1, V =VOLTRANGE, Q= 1/mpf(4), T = mpf(5)/mpf(10**3))
    B = Rfunc_CNCT(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0], Vq = A.Vq)
    B.setParameter(nterms = 400, maxA = 8, maxK = 10)
    B.genAnswer()
    plt.figure()
    plt.plot(B.rrfunction)
    plt.show()
    

        
    
        
    #    generalInput = {
#    "v":[mpf('3')* mpf(10**(4)),mpf('5.')*mpf(10**(3))],\
#    "g":[1/mpf(8),1/mpf(8)],\
#    "c":[1,1],\
#    "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.5')/mpf(10**(6)), \
#    "T" :mpf(10) / mpf(10**(3)), "Q" :1/mpf(4)}    
#    