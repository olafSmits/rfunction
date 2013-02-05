# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 08:43:43 2013
"""
from __future__ import division
import time
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
import InputParameters as BP
import cProfile
from mpl_toolkits.mplot3d import Axes3D

from numpy import newaxis, vectorize, linspace, power, arange
from sympy.mpmath import mpf, almosteq, exp
from InputParameters import GLOBAL_VOLT, GLOBAL_TEMP, EMP

mp.mp.pretty = True
mp.mp.dps = 20

fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(exp)

pi = mpf(mp.pi)
VOLTRANGE = fmfy(linspace(0,20,50)) * GLOBAL_VOLT

distance1 = linspace(1,20,50) / mpf(10**7)
distance2 = 15 / mpf(10**7)

example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
            "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
            "x":[distance1, -distance1*1.1, distance1, -distance1*1.1]}

example2 = { "v":[mpf(i)*mpf(10**j) for (i,j) in [(5,3),(5,3),(5,3),(5,3)]],
            "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
            "x":[distance2, -distance2, distance2, -distance2]}


def Rfunc_constructor(A, method = 'series'):
    if method == 'series':
        constr = Rfunc_spatial_series
    elif method =='cnct':
        constr = Rfunc_spatial_CNCT
    return constr(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0])

class Rfunc_spatial_series(object):
    T = GLOBAL_TEMP
    nterms = 250
    parameters = EMP[:,newaxis]
    g = EMP[:,newaxis]
    V = EMP[newaxis,:]
    scaledVolt = EMP[newaxis,:]
    maxParameter = EMP
    prefac = EMP
    gtot = EMP
    def __init__(self, parameters = None, maxParameter = None, distance = None,
                 g = None, V = None, prefac = None, gtot = None,
                 scaledVolt = None, T= None):
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
        self.genLDA()
        self.genGamma()
        self.mergeLDAandGamma()
        self.rfunction = self.prefac * self.lauricella
        self.rrfunction = freal(self.rfunction)

class Rfunc_spatial_CNCT(Rfunc_spatial_series):
    maxA = 10
    maxK = 9
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
        if not hasattr(self, 'wijnTerms'):
            self.genWijngaardenTerms()
        gDict = dict((k, fgamma(self.gtot + mpf(str(k)))/fgamma(self.gtot)) \
                        for k in self.wijnTerms)
        def f(i, j):
            fg = fgamma(self.scaledVolt[j])
            gamDict = dict((k, fgamma(self.scaledVolt[j] + mpf(str(k))) / \
                                (fg*gDict[k])) for k in self.wijnTerms)
            gamDict[-1] = mpf('0')
            def g(n):
                return gamDict[n]
            _g = vectorize(g)
            return _g(i)

        self.gamma = fmfy(np.ones((self.maxA, self.maxK, self.scaledVolt.size)))
        for j in xrange(self.scaledVolt.size):
            self.gamma[:,:,j] = f(self.wijngaardenArray, j)
            
    def mergeLDAandGamma(self):
        self.extractWijngaardenFromLDA()
        self.lauricella_terms =np.sum(self.lda[:,:,newaxis,:]*
                                        self.gamma[...,newaxis], axis = 1)   
        self.lauricella = levin_acceleration(self.lauricella_terms)

def levin_acceleration(L, beta = 1.):
    def LB(n, k):
        return (beta + n + k)*power(beta + n + k, k-1)/power(beta+n+k+1,k)
    def SB(n,k):
        return (beta+n+k)*(beta+n+k-1)/((beta+n+2*k)*beta+n+2*k-1)
    
    ### remainder estimator
    rem = L[1:]
    #rem = L[:-1]
    #rem = L[1:]*L[:-1]  (L[1:] - L[:-1])
    ###
    denominator = recursive_generator(1/rem, LB)
    numerator = recursive_generator(L.cumsum(axis=0)[:-1]/rem, LB)
    
    return (numerator / denominator)[0]
    
def recursive_generator(L, f):
    for i in range(1,L.shape[0]):
        L[:-1] = L[1:] - f(fmfy(arange(0,L.shape[0]-1)), i)[:,newaxis,newaxis]* L[:-1]
        L = np.delete(L,-1, axis = 0)
    return L[0,:,:]

a = BP.base_parameters(example1, V = VOLTRANGE )
A = Rfunc_constructor(a, method = 'cnct')
b = BP.base_parameters(example2, V= GLOBAL_VOLT)
B = Rfunc_constructor(b, method = 'cnct')
cProfile.runctx(
            'A.genAnswer()',
            globals(),
            locals()
        )

B.genAnswer()
def plot_surface(A):
    if not hasattr(A, 'rrfunction'):
        A.genAnswer()
        
    X = np.float64(freal(A.V))
    Y = np.float64(freal(A.distance))
    Z = np.float64(A.rrfunction)
    #return X, Y, Z
    X, Y = np.meshgrid(X,Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, np.transpose(Z))
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Distance')
    ax.set_zlabel('R function')
    plt.show()
    
