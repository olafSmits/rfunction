# -*- coding: utf-8 -*-
"""
Created on Mon Feb 04 08:43:43 2013
"""
from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
import InputParameters as BP
from Rfunc_series import Rfunc_series as Rseries
from Rfunc_cnct import Rfunc_CNCT as Rcnct
from mpl_toolkits.mplot3d import Axes3D

from numpy import vectorize, linspace
from sympy.mpmath import mpf
from InputParameters import GLOBAL_VOLT, GLOBAL_TEMP

mp.mp.pretty = True
mp.mp.dps = 20

fmfy = vectorize(mp.mpmathify)
freal = vectorize(mp.re)

pi = mpf(mp.pi)
VOLTRANGE = fmfy(linspace(0,20,25)) * GLOBAL_VOLT

distance1 = linspace(1,20,3) / mpf(10**7)
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
        constr = Rseries
    elif method =='cnct':
        constr = Rcnct
    return constr(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0])



a = BP.base_parameters(example1, V = VOLTRANGE )
A = Rfunc_constructor(a, method = 'cnct')
b = BP.base_parameters(example2, V= GLOBAL_VOLT)
B = Rfunc_constructor(b, method = 'cnct')
A.genAnswer()
#cProfile.runctx('A.genAnswer()', globals(), locals() )
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
    
