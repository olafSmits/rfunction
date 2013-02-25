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
fcos = np.vectorize(mp.cos)
fexp = np.vectorize(mp.exp)
    
def plot_for_frequency(saving = False):
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
        mp.mp.dps= 20
        A = base_parameters(pdict, V = Vpoints, Q = Qe, T = 3/mpf(10**3))
        
        B = Rfunc_constructor(A, method = 'fortran')
        B.setParameter(nterms = 100000, maxA = 20, maxK = 20)
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

def Propagator_three_modes(saving = False):
    """Plots the modulating function and interference current for an edge with
    3 modes."""
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = np.array(mpf('4.5')/ mpf(10**(6)))
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(10,3),(3,3),(5,3)]],
        "x":[dist1,  dist1, dist1],
        "g":[mpf(1)/mpf(10),mpf(1)/mpf(10), mpf(1)/mpf(10)], 
        "c":[1,1,1]}        

                            
    mp.mp.dps= 50    
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = mpf(5)/mpf(10**3))
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 800)
    _, interference = Current(B)
    
    fig = plt.figure()
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]      
    
    ax = fig.add_subplot(211)  
    ax.plot(Vpoints,interference, label = r"With interference", linewidth=1.5)                  
    
    for i in ax.get_lines(): i.set_color('black')
  
    ax.set_title(r'Tunnelling current for edge with three modes')
    ax.set_ylabel(r"$I_B/\mathrm{max}(I_B){}$")
    ax.set_ybound([0,1])
    ax.set_yticks([0,.25, .5, .75, 1])
    ax.set_yticklabels([0,0.25, 0.5, 0.75, 1])
    ax.set_xticks(xt)
    ax.set_xticklabels([])
    ax.legend(loc = 'upper right', prop={'size':12})
    plt.setp(ax.get_xticklabels(), fontsize = 12.)
    plt.setp(ax.get_yticklabels(), fontsize = 12.)    
    ax.grid(True)
    ax2 = fig.add_subplot(212)
    ax2.plot(Vpoints, B.rrfunction, linewidth=1.5) 
    ax2.get_lines()[0].set_color('black')
    ax2.set_title(r'Modulating function for edge with three modes')
    ax2.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax2.set_xlabel(r'Volt [$\mu$V]')
    ax2.set_yticks([-0.25,0,.25,.5,.75,1])
    ax2.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax2.set_xticks(xt) 
    ax2.set_xticklabels(xt_labels)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), fontsize=12.)
    plt.setp(ax2.get_yticklabels(), fontsize=12.)        
    if saving: plt.savefig('main_plot.png', bbox_inches=0, dpi=fig.dpi)
    plt.show()
    return B
    

def old_area_change(saving = False):
    """Main plot:
    Plots the mod-H function, the current with and without interference
    # and transmission"""
    Vpoints = np.array([mpf(25)/mpf(10**6),
                        mpf(50)/mpf(10**6)])
    Nterms= 100
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    dist2 = np.linspace(mpf('2.0')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), Nterms)
    
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(1,3),(1,3),(6,2),(6,2)]],
        "c":[1,1,1,1],
        "g":[mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8)],
        "x":[-dist1, dist2, -dist1, dist2]}
    mp.mp.dps= 100
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 500, maxA = 20, maxK = 20)
    B.genAnswer()
    
    return B
    
def plot_AB(B, bfield = 66.4):
    fig = plt.figure()
    ppoints= np.linspace(mpf('2.0')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), 
                     B.distance.size)
    xt = np.linspace(mpf('2.0')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), 
                     6)

    ax1 =  fig.add_subplot(311)
    ax2 =  fig.add_subplot(312)
    ax3 =  fig.add_subplot(313)
    ab_phase = area_change(B.distance.size, bfield = bfield)
    ax1.plot(ppoints,np.real(ab_phase)/np.max(np.abs(ab_phase)), linewidth=1.5)
    ax1.get_lines()[0].set_color('black')
    ax1.set_title(r'AB oscillations without dynamical interference')
    ax1.set_ylabel(r"AB phase ($e^{i\Phi}$)")
    ax1.set_ybound([-1,1])
    ax1.set_yticks([-1,-.5, 0, .5, 1])
    ax1.set_yticks([-1,-0.5, 0, 0.5, 1])
    ax1.set_xticks(xt)
    ax1.set_xticklabels([])    
    ax1.grid(True)

    dashstyle  = [[8,3], [2,1] ,[1,.1]]    
    for i in range(0,B.V.size):
        lab = str(int(B.V[i]*10**6)) +r' [$\mu$V]'
        ax2.plot(ppoints,B.rrfunction[i,:], label = lab, linewidth=1.5)
        ax2.set_title(r'Modulating function')
        ax2.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
        ax2.set_ybound([-.3,.3])
        ax2.set_yticks([-.2,-.1, 0, .1, .2])
        ax2.set_yticks([-0.2,-0.1, 0, 0.1, 0.2])
        ax2.set_xticks(xt)
        ax2.set_xticklabels([])    
        ax2.grid(True)
        ax2.get_lines()[i].set_color('black')
        ax2.get_lines()[i].set_dashes(dashstyle[i])          
        ax2.legend(loc = 'upper right')


        total_phase = np.real(np.complex128(ab_phase * np.transpose(B.rfunction[i,:])))
        
        
        ax3.plot(ppoints,total_phase / np.max(np.abs(total_phase)), linewidth=1.5)
        ax3.set_title(r'Combined interference signal (normalized)')
        ax3.set_ylabel(r"$\mathrm{Re}[e^{i\Phi}H_{ij}^{\mathrm{mod}}]{}$")
        ax3.set_ybound([-1,1])
        ax3.set_yticks([-1,-.5, 0, .5, 1])
        ax3.set_yticks([-1,-0.5, 0, 0.5, 1])
        ax3.set_xticks(xt)
        ax3.set_xticklabels(10**6 * np.linspace(mpf('2.0')/mpf(10**(6)),
                                                mpf('3.0')/mpf(10**(6)),6))            
        ax3.grid(True)
        
        
        ax3.set_xlabel(r'distance (one edge) [$\mu$m]')        
        ax3.xaxis.set_label_coords(.94, -0.13)

        ax3.get_lines()[i].set_color('black')
    
        ax3.get_lines()[i].set_dashes(dashstyle[i])          
        
    plt.show()    
    
def change_distance_laughlin(saving = False):
    """Main plot:
    Plots the mod-H function, the current with and without interference
    # and transmission"""
    Vpoints = np.array([mpf(25)/mpf(10**6),
                        mpf(50)/mpf(10**6)])
    Nterms= 100
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    dist2 = np.linspace(mpf('2.')/mpf(10**(6)), mpf('3')/mpf(10**(6)), Nterms)
    
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5.4,2),(5.4,2)]],
        "c":[1,1],
        "g":[mpf(1)/mpf(2),mpf(1)/mpf(2)],
        "x":[-dist1, dist2]}
    mp.mp.dps= 100
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 500, maxA = 20, maxK = 20)
    #B.genAnswer()
    
    return B    

    
def area_change(Nterms = 40, bfield = 67):
    phase = 2*mp.pi * bfield * ELEC / (2*2*mp.pi*HBAR *mpf(10**12))
    def f(x):
        return fexp(1j*phase*x+ 1j*mp.pi/4.)
    
    return np.complex128(f(np.linspace(0,.1,Nterms)))

