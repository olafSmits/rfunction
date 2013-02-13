# -*- coding: utf-8 -*-
"""
Created on Wed Feb 06 18:22:52 2013
"""
from __future__ import division
import Rfunc_cnct as Rcnct
import numpy as np
import sympy.mpmath as mp
from cnct import cnct as cnct
import matplotlib.pylab as plt
from numpy import newaxis, vectorize
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
        self.gamma = self.gamma
        self.lda = fmfy(self.lda)
        self.lauricella_terms =np.asfortranarray(np.sum(self.lda[:,:,newaxis,:]*
                                        self.gamma[...,newaxis], axis = 1))
                                        
        #shape of lauricella_terms is (maxA, V, distance)
        self.lauricella_terms = cnct.levin_acceleration(self.lauricella_terms, 1.)
        self.lauricella = self.lauricella_terms[0,:,:]
        
if __name__ == '__main__':
    import InputParameters as BP
    VOLTRANGE = fmfy(np.linspace(0,200,50)) * mpf(1)/ mpf(10**6)
#    basedist = mpf(1.0)/mpf(10**6)
#    distance = np.linspace(.001, 1.0, 40) * basedist
#    distance2 = np.ones_like(distance) * basedist
#    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
#              "c":[1,1,1,1],
#            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
#                 "x":[distance2, -distance, distance2, -distance]}
#    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3)]],
#              "c":[1,1],
#            "g":[1/mpf(8),1/mpf(8)],
#                 "x":[distance2, -distance]}
#    A = BP.base_parameters(example1, V =VOLTRANGE, Q= 1/mpf(4), T = 5/ mpf(10**3))
#    B = Rfunc_fortran(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
#                                maxParameter = A.maxParameter, prefac = A.prefac,
#                                V = A.V, scaledVolt = A.scaledVolt,
#                                distance = A.input_parameters["x"][0], Vq = A.Vq)
#    B.setParameter(nterms = 1500, maxA = 15, maxK = 15)
#    B.genAnswer()
#    plt.figure()
#    plt.plot(B.rrfunction)
#    plt.show()
    
    basedist = mpf(1.5)/mpf(10**6)
    distance = np.linspace(.8, 1.2, 3) * basedist
    distance2 = np.ones_like(distance) * basedist
    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(3,4),(3,4),(5,3),(5,3)]],
                 "c":[1,1,1,1],
                 "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
                 "x":[distance2, -distance, distance2, -distance]}
    A = BP.base_parameters(example1, V =VOLTRANGE, Q= 1/mpf(4), T = mpf(20)/10**3 )
    B = Rfunc_fortran(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0], Vq = A.Vq)
    B.setParameter(nterms = 200000, maxA = 15, maxK = 15)
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