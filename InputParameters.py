# -*- coding: utf-8 -*-
"""
Created on Tue Feb 05 17:04:52 2013
"""
from pprint import pprint
import numpy as np
import sympy.mpmath as mp

from numpy import newaxis, vectorize
from sympy.mpmath import mpf, mpc, exp

fmfy = vectorize(mp.mpmathify)
fexp = np.vectorize(exp)
fmpc = vectorize(mpc)

BLTZMN = mpf('1.3806403') /mpf(10**(23))
HBAR = mpf('1.054571628') /mpf(10**(34))
pi = mpf(mp.pi)
ELEC = mpf('1.60217646')/mpf(10**(19))
TOL = 1 / mpf(10**20)

GLOBAL_TEMP = mpf(5) / mpf(10**3)
GLOBAL_VOLT = mpf(1) / mpf(10**6)
EMP= np.array([])



class base_parameters(object):
    input_parameters = { "v":EMP,"c":EMP,"g":EMP,"x":EMP}
    def __init__(self, input_parameters, V, Q, T):
        self.V, self.Q, self.T = V, mpf(Q), mpf(T)
########## Process input # Perform Sanity Checks ###########
        for i, j in input_parameters.items():
            if not isinstance(j, list):
                raise ValueError
            self.input_parameters[i] = np.asarray(j)
        if len(set(map(len, input_parameters.values()))) > 1:
            raise ValueError
        if not isinstance(self.V, np.ndarray) or not isinstance(self.V, list):
            self.V = np.array([self.V]).ravel()
        self.isZeroT = mp.almosteq(self.T, mpf(0))
########### Restructure distance and voltage, if necessary
        if len(self.input_parameters["x"].shape) == 1:
            self.input_parameters["x"] = self.input_parameters["x"][:,newaxis]
########### Generate parameters
        self.gtot = mp.fsum(self.input_parameters["g"])
        self.genParameters()
        self.genScaledVoltage()
    def genParameters(self):
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
        self.genPrefactor()
        
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
                    
            # compute exponentiated parameters, z_i = exp(x_i/v_i)
            new_line = self.getNextLine(new_line)
            maxP, argm = np.max(new_line), np.argmax(new_line)
            
            #remove largest parameter, z_max
            maxParameter.append(maxP)
            new_line = np.delete(new_line, argm)
            new_g_line = np.delete(new_g_line, argm)
            
            # final parameters: 1- z_i/z_max
            new_line = self.getParameter(new_line, maxP)
            
            new_parameters.append(new_line)
            new_g.append(new_g_line)
        
        self.maxParameter = np.array(maxParameter)
        if self.isZeroT:
            self.maxParameter = fexp(self.maxParameter)
        self.parameters = np.array(new_parameters)
        self.g = np.array(new_g)
        # Note: self.parameters and self.g has as alements: arrays
        # for each value of inputparameters a possible reduction can take place.

    def genScaledVoltage(self):
        if self.isZeroT:
            self.Vq = self.Q*ELEC*self.V / HBAR
            self.scaledVolt = -1j*self.Vq
        else:
            self.Vq = self.Q*ELEC*self.V/(2*pi*BLTZMN*self.T)        
            self.scaledVolt = fmpc(self.gtot /mpf(2), - self.Vq )
    def genPrefactor(self):
        if self.isZeroT:
            self.prefac = fmfy(np.ones_like(self.parameters[:,0]))
        else:
            fac = 2*pi*BLTZMN*self.T / HBAR        
            self.prefac = fexp(np.sum(self.g * self.parameters, axis = 1)*fac/mpf(2))
    def getNextLine(self, new_line):
        if self.isZeroT:
            return new_line
        else:
            fac = 2*pi*BLTZMN*self.T / HBAR 
            return fexp(fac*new_line)
    def getParameter(self, new_line, maxP):
        if self.isZeroT:
            return maxP - new_line
        else:
            return 1- new_line / maxP


if __name__ == '__main__':
    Vpoints = mp.linspace(0, mpf('2.')/mpf(10**4), 201)
    dist1 = mpf('1.7')/ mpf(10**(6))
    dist2 = mpf('1.5')/mpf(10**(6))
    genData = { 
         "v":[mpf(i) * mpf(10**j) for (i,j) in [(3,4),(3,4),(5,3),(5,3)]],
         "c":[1,1,1,1],
         "g":[1/mpf(8), 1/mpf(8), 1/mpf(8), 1/mpf(8)],
         "x":[dist1, -dist2, dist1, -dist2]}
    
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)