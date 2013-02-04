# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 08:43:43 2013
"""
import time
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
from numpy import newaxis, vectorize, linspace
from sympy.mpmath import mpf, mpc, almosteq, exp

mp.mp.pretty = True
mp.mp.dps = 20
fmathify = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
fexp = np.vectorize(exp)
freal = vectorize(mp.re)
fmpc = vectorize(mpc)

BLTZMN = mpf('1.3806403') /mpf(10**(23))
HBAR = mpf('1.054571628') /mpf(10**(34))
pi = mpf(mp.pi)
ELEC = mpf('1.60217646')/mpf(10**(19))
TOL = 1 / mpf(10**20)

GLOBAL_TEMP = mpf(5) / mpf(10**3)
GLOBAL_VOLT = mpf(1) / mpf(10**6)
VOLTRANGE = fmathify(linspace(0,10,35)) * GLOBAL_VOLT
EMP= np.array([])
distance1 = linspace(1,15,30) / mpf(10**7)
distance2 = 15 / mpf(10**7)

example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,2),(5,2),(1,3),(1,3)]],
            "c":[1,1,1,1],
            "g":[1/mpf(3),1/mpf(3),1/mpf(8),1/mpf(8)],
            "x":[distance1, -distance1*1.1, distance1, -distance1*1.1]}

example2 = { "v":[mpf(i)*mpf(10**j) for (i,j) in [(5,3),(5,3),(5,3),(5,3)]],
            "c":[1,1,1,1],
            "g":[1/mpf(3),1/mpf(3),1/mpf(8),1/mpf(8)],
            "x":[distance2, -distance2, distance2, -distance2]}

class base_parameters(object):
    T = GLOBAL_TEMP
    V = [GLOBAL_VOLT, GLOBAL_VOLT*10]
    Q =  1/mpf(4)
    input_parameters = { "v":EMP,"c":EMP,"g":EMP,"x":EMP}
    def __init__(self, input_parameters, V = None, Q = None, T = None):
########## Process input
        for i, j in input_parameters.items():
            if isinstance(j, np.ndarray):         
                self.input_parameters[i] = j
            else:
                self.input_parameters[i] = np.asarray(j)
        if len(set(map(len, input_parameters.items()))) > 1:
            raise ValueError
        if T is not None:
            self.T = T
        if V is not None:
            self.V = V
        if Q is not None:
            self.Q = Q
        if not isinstance(self.V, np.ndarray) or not isinstance(self.V, list):
            self.V = np.array([self.V]).ravel()
########### Restructure distance and voltage, if necessary
        if len(self.input_parameters["x"].shape) == 1:
            self.input_parameters["x"] = self.input_parameters["x"][:,newaxis]
########### Generate parameters
        self.gtot = mp.fsum(self.input_parameters["g"])
        self.genParameters()
        self.genScaledVoltage()
    def genParameters(self):
        
        ####### 
        v = self.input_parameters["v"]
        x = self.input_parameters["x"]
        c = self.input_parameters["c"]
        g = self.input_parameters["g"]

        def apply_take(a, s):
            m, n = a.shape
            return np.transpose(np.take(np.transpose(a), s + np.mgrid[0:m,0:n][1] *m ))
        # Sort self.parameters
        # Sort self.g accordingly
        self.parameters = c[:,newaxis] * x / v[:,newaxis]
        m, n = self.parameters.shape
        self.g = np.transpose(np.vstack([[g]]*n))
        
        sort_ind = np.argsort(self.parameters, axis = 0)        
        self.parameters = apply_take(self.parameters, sort_ind)
        self.g = apply_take(self.g, sort_ind)

        # Determine prefactor        
        fac = 2*pi*BLTZMN*self.T / HBAR        
        self.prefac = fexp(np.sum(self.g * self.parameters, axis = 1)*fac/mpf(2))
        
        # Remove duplicates in self.parameter
        # Adjust self.g accordingly
        # Example: self.parameter[i] = self.parameter[i+1]
        # Then we set: self.g[i] = self.g[i] + self.g[i+1]
        # and self.parameter[i+1] and self.g[i+1] are deleted afterwards
        difference = np.diff(self.parameters, axis = 1)
        
        new_parameters = []
        maxParameter = []
        new_g = []
        for i in xrange(n):
            new_line = self.parameters[i,:]
            new_g_line = self.g[i,:]
            for j in xrange(m-2,-1,-1):
                if difference[i,j] < TOL:       
                    new_line = np.delete(new_line,j+1)
                    new_g_line[j] += new_g_line[j+1]
                    new_g_line = np.delete(new_g_line, j+1)
            new_line = fexp(fac*new_line)
            maxP, argm = np.max(new_line), np.argmax(new_line)
            maxParameter.append(maxP)
            new_line = np.delete(new_line, argm)
            new_g_line = np.delete(new_g_line, argm)
            
            new_line = 1- new_line / maxP
            
            new_parameters.append(new_line)
            new_g.append(new_g_line)

        self.maxParameter = np.array(maxParameter)
        self.parameters = np.array(new_parameters)
        self.g = np.array(new_g)
        # Note: self.parameters and self.g has as alements: arrays
        # for each value of inputparameters a possible reduction can take place.


        # Now we create the physical values


#        self.not_select = np.insert(diff, 0, mpf(1)) < TOL
#        self.select_ind = np.append(diff, mpf(1)) > TOL
#        self.parameters = self.parameters[sort_ind]
#        self.s = select_ind
#        g = g[sort_ind] + g[sort_ind] * not_select[:,newaxis]
#        self.g = g[select_ind]
#        
        #self.parameters = self.parameters[sort_ind][select_ind]
        #g = g[sort_ind][select_ind]
        #self.g = g

    def genScaledVoltage(self):
        self.scaledVolt = fmpc(self.gtot, 
                               -self.Q*ELEC*self.V/(2*pi*BLTZMN*self.T) )

def Rfunc_constructor(A):
    return Rfunc_spatial_series(parameters = A.parameters, g = A.g, gtot = A.gtot,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                T = A.T, V = A.V, scaledVolt = A.scaledVolt)

class Rfunc_spatial_series(object):
    T = GLOBAL_TEMP
    nterms = 100
    parameters = EMP[:,newaxis]
    g = EMP[:,newaxis]
    V = EMP[newaxis,:]
    scaledVolt = EMP[newaxis,:]
    maxParameter = EMP
    prefac = EMP
    gtot = EMP
    def __init__(self, parameters = None, maxParameter = None,
                 g = None, V = None, prefac = None, gtot = None,
                 scaledVolt = None, T= None):
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
        self.genAnswer()
    def genLDA(self):
        self.power = np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:])
        self.ts = np.sum(self.g[...,newaxis] * 
                        np.power(self.parameters[...,newaxis], 
                         np.arange(1,self.nterms)[newaxis,newaxis,:]),axis = 1)
        self.lda = np.ones((self.nterms, self.parameters.shape[0]))
        self.lda = fmathify(self.lda)
        self.ts = np.transpose(self.ts)
        for m in xrange(1,self.nterms):
            self.lda[m,:] = np.sum(self.lda[:m,:][::-1] * self.ts[:m,:], axis=0)/mpf(m)
    def genGamma(self):
        self.gamma = np.arange(0, self.nterms)[:, newaxis] + self.scaledVolt[newaxis,:]
        self.gamma = fgamma(self.gamma) / fgamma(self.scaledVolt)
        self.div = fgamma(self.gtot + np.arange(0, self.nterms)) / fgamma(self.gtot)
        self.gamma = self.gamma / self.div[:,newaxis]
    def mergeLDAandGamma(self):
        self.lauricella_terms = self.lda[:,newaxis,:] * self.gamma[:,:,newaxis]
        self.lauricella = np.sum(self.lauricella_terms, axis =0)
    def genAnswer(self):
        self.genLDA()
        self.genGamma()
        self.mergeLDAandGamma()
        self.rfunction = self.prefac * self.lauricella
        self.rrfunction = freal(self.rfunction)


a = base_parameters(example1, V = VOLTRANGE)
A = Rfunc_constructor(a)
b = base_parameters(example2, V= GLOBAL_VOLT)
B = Rfunc_constructor(b)
