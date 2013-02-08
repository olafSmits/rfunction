# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 11:35:26 2013
"""

from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt

import InputParameters as BP
from Rfunc_series import Rfunc_series as Rseries
from Rfunc_cnct import Rfunc_CNCT as Rcnct
from Rfunc_cnct_fortran import Rfunc_fortran as Rfortran

from mpl_toolkits.mplot3d import Axes3D
from numpy import vectorize, linspace, newaxis
from sympy.mpmath import mpf


mp.mp.pretty = True
mp.mp.dps = 20

fbeta = vectorize(mp.beta)
fsinh = vectorize(mp.sinh)
fmfy = vectorize(mp.mpmathify)
freal = vectorize(mp.re)

pi = mpf(mp.pi)



def Rfunc_constructor(A, method = 'series'):
    if method == 'series':
        constr = Rseries
    elif method =='cnct':
        constr = Rcnct
    elif method == 'fortran':
        constr = Rfortran
    else:
        raise ValueError
    return constr(parameters = A.parameters, g = A.g, gtot = A.gtot, T = A.T,
                                maxParameter = A.maxParameter, prefac = A.prefac,
                                V = A.V, scaledVolt = A.scaledVolt,
                                distance = A.input_parameters["x"][0], Vq = A.Vq)


def Current(rfunc):
    """
    Takes an rfunc object and calculates the single and interference current.
    rfunc object can be of type fortran, series or cnct.
    """
    gtot = rfunc.gtot
    Vscaled = rfunc.scaledVolt
    Vq = rfunc.Vq
    
    single = fsinh(Vq * pi) * fbeta(Vscaled, gtot - Vscaled)
    if not hasattr(rfunc, 'rrfunction'):
        rfunc.genAnswer()
    interference = (1+2*rfunc.rrfunction) * single[:,newaxis]
    single = freal(single)
    interference = freal(interference)
    return single, interference

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
    
if __name__ == '__main__':
    VOLTRANGE = fmfy(np.linspace(0,50,3)) * BP.GLOBAL_VOLT
    basedist = mpf(1.0)/mpf(10**6)
    distance = np.linspace(.5, 1.0, 3) * basedist
    distance2 = np.ones_like(distance) * basedist
    example1 = { "v":[mpf(i) * mpf(10**j) for (i,j) in [(2,3),(2,3),(8,3),(8,3)]],
              "c":[1,1,1,1],
            "g":[1/mpf(8),1/mpf(8),1/mpf(8),1/mpf(8)],
                 "x":[distance2, -distance, distance2, -distance]}
    A = BP.base_parameters(example1, V =VOLTRANGE)
    B = Rfunc_constructor(A, 'fortran')
    B.setParameter(nterms = 800, maxA = 8, maxK = 10)
    B.genAnswer()
    single, interference = Current(B)
#    plt.figure()
#    plt.plot(B.rrfunction)
#    plt.show()