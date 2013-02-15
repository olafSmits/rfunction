# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:29:21 2013
"""

from __future__ import division
import sympy.mpmath as mp
from sympy.mpmath import mpf
import time as time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft

import os
mp.mp.dps= 20
mp.mp.pretty = True

from Interface import Rfunc_constructor, Current
from InputParameters import base_parameters, ELEC, HBAR

_mp = np.vectorize(mp.mpf)
    
def temperature(saving = False):
    Vpoints = mp.linspace(0, mpf('3.')/mpf(10**4), 401)
    dist1 = np.array(mpf('2.4')/ mpf(10**(6)))
    dist2 = np.array(mpf('1.0')/mpf(10**(6)))
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(8,3),(8,3),(2,3),(2,3)]],
         "x":[dist1, -dist2, dist1, -dist2]}
    LaughlinE3 = {
        "g":[mpf(1)/mpf(2),mpf(1)/mpf(2)],
        "c":[1,1],
        "x":[dist1, -dist2],
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(9,3),(9,3)]],"Q":1/mpf(3)}
    aBS23 = {
        "g":[mpf(1)/mpf(3), mpf(1)/mpf(3),mpf(5)/mpf(8), mpf(5)/mpf(8)], 
        "c":[1,1,1,1],"Q":1/mpf(3)}
    BS13 = {
        "g":[mpf(1)/mpf(3),mpf(1)/mpf(3),mpf(3)/mpf(8), mpf(3)/mpf(8)], 
        "c":[1,1,-1,-1],"Q":1/mpf(3)}
    aRR4 = {
        "g":[mpf(1)/mpf(12), mpf(1)/mpf(12), mpf(1)/mpf(4), mpf(1)/mpf(4)], 
        "c":[1,1,1,1],"Q":1/mpf(6)}
    Pfaff = {
        "g":[mpf(1)/mpf(10), mpf(1)/mpf(10), mpf(1)/mpf(8), mpf(1)/mpf(8)], 
        "c":[1,1,1,1],"Q":1/mpf(4)}        
        
    particleset = [aBS23, BS13, Pfaff, LaughlinE3]
    ans = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in particleset[2:3]:
        pdict = genData.copy()
        pdict.update(i)
        Qe = pdict["Q"]
        del pdict["Q"]
        mp.mp.dps= 100
        A = base_parameters(pdict, V = Vpoints, Q = Qe, T = 0)
        
        B = Rfunc_constructor(A, method = 'series')
        B.setParameter(nterms = 800, maxA = 12, maxK = 12)
        B.genAnswer()
        ans.append([A,B])
        ax.plot(Vpoints, B.rrfunction, linewidth=1.5) 
                            
    dashstyle  = [(None, None), [7,2], [3,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i])
    xt = np.linspace(0, 2 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]        
    ax.set_ylabel(r"H_mod")
    ax.set_xlabel(r'Volt')
    ax.set_yticks([-0.25,0,.25,.5,.75,1])
    ax.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.grid(True)
    ax.legend(loc='upper right', prop={'size':14})
    plt.setp(ax.get_xticklabels(), fontsize=12.)
    plt.setp(ax.get_yticklabels(), fontsize=12.)
    plt.show()
    return ans
