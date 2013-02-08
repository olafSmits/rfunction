# -*- coding: utf-8 -*-
"""
Created on Tue Feb 05 17:04:52 2013
"""

import numpy as np
import sympy.mpmath as mp

from numpy import newaxis, vectorize
from sympy.mpmath import mpf, mpc, exp

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
                    
            # compute exponentiated parameters, z_i = exp(x_i/v_i)
            new_line = fexp(fac*new_line)
            maxP, argm = np.max(new_line), np.argmax(new_line)
            
            #remove largest parameter, z_max
            maxParameter.append(maxP)
            new_line = np.delete(new_line, argm)
            new_g_line = np.delete(new_g_line, argm)
            
            # final parameters: 1- z_i/z_max
            new_line = 1- new_line / maxP
            
            new_parameters.append(new_line)
            new_g.append(new_g_line)

        self.maxParameter = np.array(maxParameter)
        self.parameters = np.array(new_parameters)
        self.g = np.array(new_g)
        # Note: self.parameters and self.g has as alements: arrays
        # for each value of inputparameters a possible reduction can take place.

    def genScaledVoltage(self):
        self.Vq = self.Q*ELEC*self.V/(2*pi*BLTZMN*self.T)        
        self.scaledVolt = fmpc(self.gtot /mpf(2), - self.Vq )
        
