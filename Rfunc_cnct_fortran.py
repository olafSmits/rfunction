# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 18:22:52 2013
"""
from __future__ import division
import Rfunc_cnct as Rcnct
import numpy as np
import sympy.mpmath as mp
import cnct as cnct


from numpy import newaxis, vectorize, power, arange
from sympy.mpmath import mpf

mp.mp.pretty = True
mp.mp.dps = 20

fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)



class Rfunc_fortran(Rcnct.Rfunc_spatial_CNCT):
    
    def genLDA(self):
        self.power = np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:])
        self.ts = np.sum(self.g[...,newaxis] * 
                        np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:]),axis = 1)
        self.lda = np.ones((self.nterms, self.parameters.shape[0]), dtype = np.float64)
        self.ts = np.float64(np.transpose(self.ts))
        
        cnct.ldagenerator(self.lda, self.ts)
        
    def mergeLDAandGamma(self):
        self.extractWijngaardenFromLDA()
        self.gamma = np.complex128(self.gamma)
        self.lda = np.complex128(self.lda)
        self.lauricella_terms =np.sum(self.lda[:,:,newaxis,:]*
                                        self.gamma[...,newaxis], axis = 1)  
                                        
        #shape of lauricella_terms is (maxA, V, distance)
        self.lauricella = cnct.levin_acceleration(self.lauricella_terms)
