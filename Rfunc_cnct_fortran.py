# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 18:22:52 2013
"""
from __future__ import division
import Rfunc_cnct as Rcnct
import numpy as np
import sympy.mpmath as mp
from cnct import cnct as cnct


from numpy import newaxis, vectorize, power, arange
from sympy.mpmath import mpf

mp.mp.pretty = True
mp.mp.dps = 20

fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)



class Rfunc_fortran(Rcnct.Rfunc_CNCT):
    
    def genLDA(self):
        self.power = np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:])
        self.ts = np.sum(self.g[...,newaxis] * 
                        np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:]),axis = 1)
        self.lda = np.ones((self.nterms, self.parameters.shape[0]), dtype = np.float64)
        self.ts = np.float64(np.transpose(self.ts))
        
        self.lda = cnct.genlda(self.lda, self.ts)
        
    def mergeLDAandGamma(self):
        self.extractWijngaardenFromLDA()
        self.gamma = np.complex128(self.gamma)
        self.lda = np.complex128(self.lda)
        self.lauricella_terms =np.asfortranarray(np.sum(self.lda[:,:,newaxis,:]*
                                        self.gamma[...,newaxis], axis = 1))
                                        
        #shape of lauricella_terms is (maxA, V, distance)
        self.lauricella_terms = cnct.levin_acceleration(self.lauricella_terms, 1.)
        self.lauricella = self.lauricella_terms[0,:,:]
        
if __name__ == '__main__':
    import InputParameters as BP
    VOLTRANGE = fmfy(np.linspace(0,100,50)) * BP.GLOBAL_VOLT
    basedist = mpf(1.0)/mpf(10**6)
    distance = np.arange(0,3) * basedist
    distance2 = np.ones_like(distance) * basedist
    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
              "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
                 "x":[distance2, -distance, distance2, -distance]}
    A = BP.base_parameters(example1, V =VOLTRANGE)
    B = Rfunc_fortran(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0])
    B.setParameter(nterms = 10000, maxA = 15, maxK = 15)