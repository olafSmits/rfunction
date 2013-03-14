# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 17:12:47 2013
"""

from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
import time
from numpy import newaxis, vectorize
from sympy.mpmath import mpf
from InputParameters import GLOBAL_TEMP, EMP


fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)

class Rfunc_series(object):
    nterms = 250
    parameters = EMP[:,newaxis]
    g = EMP[:,newaxis]
    V = EMP[newaxis,:]
    scaledVolt = EMP[newaxis,:]
    maxParameter = EMP
    prefac = EMP
    gtot = EMP
    def __init__(self, parameters, maxParameter, 
                 g, V, prefac, gtot,
                 scaledVolt, T, Vq):
        self.Vq = Vq
        self.parameters = parameters
        self.V = V
        self.T = T
        self.isZeroT = mp.almosteq(self.T, mpf(0))
        self.scaledVolt = scaledVolt
        self.g = g
        self.maxParameter = maxParameter
        self.prefac = prefac
        self.gtot = gtot
        ### input check ###            
        if len(self.maxParameter.shape) == 1:
            self.maxParameter = self.maxParameter[newaxis,:]
        if len(self.prefac.shape) == 1:
            self.prefac = self.prefac[newaxis,:]
        if self.g.shape != self.parameters.shape or \
                    self.V.shape != self.scaledVolt.shape or \
                    self.maxParameter.shape[1] != self.parameters.shape[0]:
            
            raise ValueError
        if len(self.g.shape) == 1:
            self.g = self.g[:, newaxis]
            self.parameters = self.parameters[:, newaxis]
        self.prefac = self.prefac * np.power(self.maxParameter, 
                                  -self.scaledVolt[...,newaxis])
    def setParameter(self, nterms, **arg):
        self.nterms = nterms
    def genLDA(self):
        self.power = np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:])
        self.ts = np.sum(self.g[...,newaxis] * 
                        np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:]),axis = 1)
        self.lda = np.ones((self.nterms, self.parameters.shape[0]))
        self.lda = fmfy(self.lda)
        self.ts = np.transpose(self.ts)
        for m in xrange(1,self.nterms):
            self.lda[m,:] = np.sum(self.lda[:m,:][::-1] *\
                                        self.ts[:m,:], axis=0)/mpf(m)
    def genGamma(self):
        self.gamma = self.__genGammaTerms()
        div = fgamma(self.gtot + np.arange(0, self.nterms)) / fgamma(self.gtot)
        self.gamma = self.gamma / div[:,newaxis]
    def mergeLDAandGamma(self):
        self.lauricella_terms = self.lda[:,newaxis,:] * self.gamma[:,:,newaxis]
        self.lauricella = np.sum(self.lauricella_terms, axis =0)
    def genAnswer(self):
        #cProfile.runctx('self.genLDA()', globals(), locals() )
        #cProfile.runctx('self.genGamma()', globals(), locals() )
        t1 = time.time()
        self.genLDA()
        self.genGamma()
        self.mergeLDAandGamma()
        #cProfile.runctx('self.mergeLDAandGamma()', globals(), locals() )
        self.rfunction = self.prefac * self.lauricella
        self.rrfunction = freal(self.rfunction)
        t2 = time.time()
        t3 = np.round((t2-t1), decimals =1)
        print "R function computed (g = " + str(self.gtot) + "). Took: " + \
                str(t3) + " seconds."
    def __genGammaTerms(self):
        if self.isZeroT:
            _gam = np.arange(0, self.nterms)[:, newaxis]
            return np.power(self.scaledVolt[newaxis,:], _gam)
        else:
            _gam = np.arange(0, self.nterms)[:, newaxis] + \
                                    self.scaledVolt[newaxis,:]
            return fgamma(_gam) / fgamma(self.scaledVolt)

            
class from_hypergeometric(Rfunc_series):
    def genAnswer(self):

        if self.isZeroT:
            p = -1j*(self.scaledVolt[:,newaxis] * np.transpose(self.parameters))/2
            g = self.gtot/2-1/mpf(2)
            def _f(x):
                return mp.gamma(g+1)* \
                np.power(x/mpf(2), -g) * \
                mp.exp(1j*x) * mp.besselj(g, x)
            __f = np.vectorize(_f)
            self.lauricella = __f(p)
        else:                
            _hyp2f1 = np.vectorize(mp.hyp2f1)
            self.lauricella = _hyp2f1(self.scaledVolt[:,newaxis], self.gtot/2,\
                                self.gtot, np.transpose(self.parameters))


        if len(self.lauricella.shape)== 1:
            self.rfunction = self.prefac * self.lauricella[:,newaxis]
        else:
            self.rfunction = self.prefac * self.lauricella
            
        self.rrfunction = freal(self.rfunction)

        
if __name__ == '__main__':
    import InputParameters as BP
    VOLTRANGE = fmfy(np.linspace(0,200,50)) * mpf(1)/ mpf(10**6)
#    basedist = mpf(1.0)/mpf(10**6)
#    distance = np.linspace(0.1, 1.0, 5) * basedist
#    distance2 = np.ones_like(distance) * basedist
#    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
#              "c":[1,1,1,1],
#            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
#                 "x":[distance2, -distance, distance2, -distance]}
#    A = BP.base_parameters(example1, V =VOLTRANGE, Q = 1/mpf(4), T = 1 / mpf(10**4))
#    B = Rfunc_series(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
#                                maxParameter = A.maxParameter, prefac = A.prefac,
#                                V = A.V, scaledVolt = A.scaledVolt,
#                                distance = A.input_parameters["x"][0], Vq = A.Vq)
#    B.setParameter(nterms = 200)
#    B.genAnswer()
#    plt.figure()
#    plt.plot(B.rrfunction)
#    plt.show()
    
#    basedist = mpf(1.5)/mpf(10**6)
#    distance = np.linspace(.8, 1.2, 3) * basedist
#    distance2 = np.ones_like(distance) * basedist
#    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(3,4),(3,4),(5,3),(5,3)]],
#                 "c":[1,1,1,1],
#                 "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
#                 "x":[distance2, -distance, distance2, -distance]}
#    A = BP.base_parameters(example1, V =VOLTRANGE, Q= 1/mpf(4), T = 0 )
#    B = Rfunc_series(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
#                                maxParameter = A.maxParameter, prefac = A.prefac,
#                                V = A.V, scaledVolt = A.scaledVolt,
#                                distance = A.input_parameters["x"][0], Vq = A.Vq)
#    B.setParameter(nterms = 200, maxA = 8, maxK = 10)
#    B.genAnswer()
#    plt.figure()
#    plt.plot(B.rrfunction)
#    plt.show()
    
#===============================================================================
# 
#===============================================================================
    Vpoints = mp.linspace(0, mpf('2.')/mpf(10**4), 201)
    dist1 = np.array([mpf('1.7') / mpf(10**(6)), mpf('1.7')/ mpf(10**(6))])
    dist2 = np.array([mpf('1.5') / mpf(10**(6)), mpf('1.7') / mpf(10**(6))])
    genData = { 
         "v":[mpf(i) * mpf(10**j) for (i,j) in [(3,4),(3,4)]],
         "c":[1,1],
         "g":[1/mpf(8), 1/mpf(8)],
         "x":[dist1, -dist2]}
    
    A = BP.base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_series(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0], Vq = A.Vq)
    B.genAnswer()
    plt.figure()
    plt.plot(B.rrfunction)
    plt.show()                                