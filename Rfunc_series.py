# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 17:12:47 2013
"""

from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
from numpy import newaxis, vectorize
from sympy.mpmath import mpf
from InputParameters import GLOBAL_TEMP, EMP


fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)

class Rfunc_series(object):
    T = GLOBAL_TEMP
    nterms = 250
    parameters = EMP[:,newaxis]
    g = EMP[:,newaxis]
    V = EMP[newaxis,:]
    scaledVolt = EMP[newaxis,:]
    maxParameter = EMP
    prefac = EMP
    gtot = EMP
    def __init__(self, parameters, maxParameter, distance,
                 g, V, prefac, gtot,
                 scaledVolt, T, Vq):
        if Vq is not None:
            self.Vq = Vq
        if distance is not None:
            self.distance = distance
        if parameters is not None:
            self.parameters = parameters
        if V is not None:
            self.V = V
        if T is not None:
            self.T = T
        if scaledVolt is not None:
            self.scaledVolt = scaledVolt
        if g is not None:
            self.g = g
        if maxParameter is not None:
            self.maxParameter = maxParameter
        if prefac is not None:
            self.prefac = prefac
        if gtot is not None:
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
        self.gamma = np.arange(0, self.nterms)[:, newaxis] + \
                                    self.scaledVolt[newaxis,:]
        self.gamma = fgamma(self.gamma) / fgamma(self.scaledVolt)
        div = fgamma(self.gtot + np.arange(0, self.nterms)) / fgamma(self.gtot)
        self.gamma = self.gamma / div[:,newaxis]
    def mergeLDAandGamma(self):
        self.lauricella_terms = self.lda[:,newaxis,:] * self.gamma[:,:,newaxis]
        self.lauricella = np.sum(self.lauricella_terms, axis =0)
    def genAnswer(self):
        #cProfile.runctx('self.genLDA()', globals(), locals() )
        #cProfile.runctx('self.genGamma()', globals(), locals() )
        
        self.genLDA()
        self.genGamma()
        self.mergeLDAandGamma()
        #cProfile.runctx('self.mergeLDAandGamma()', globals(), locals() )
        self.rfunction = self.prefac * self.lauricella
        self.rrfunction = freal(self.rfunction)
        
if __name__ == '__main__':
    import InputParameters as BP
    VOLTRANGE = fmfy(np.linspace(0,50,3)) * BP.GLOBAL_VOLT
    basedist = mpf(1.0)/mpf(10**6)
    distance = np.linspace(.5, 1.0, 3) * basedist
    distance2 = np.ones_like(distance) * basedist
    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
              "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
                 "x":[distance2, -distance, distance2, -distance]}
    A = BP.base_parameters(example1, V =VOLTRANGE)
    B = Rfunc_series(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0], Vq = A.Vq)
    B.setParameter(nterms = 800)
    B.genAnswer()
    plt.figure()
    plt.plot(B.rrfunction)
    plt.show()