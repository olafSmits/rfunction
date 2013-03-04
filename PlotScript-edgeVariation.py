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
    
    


def area_change(saving = False):
    """Main plot:
    Plots the mod-H function, the current with and without interference
    # and transmission"""
    Vpoints = np.array([
                        mpf(60)/mpf(10**6)])
    Nterms= 151
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    dist2 = np.linspace(mpf('0.5')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), Nterms)
    
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(1,3),(1,3),(6,2),(6,2)]],
        "c":[1,1,1,1],
        "g":[mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8)],
        "x":[-dist1, dist2, -dist1, dist2]}
    mp.mp.dps= 150
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 700, maxA = 20, maxK = 20)
    B.genAnswer()
    
    return A,B
    
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

    
def ABOscillations(PlotPoints = 40, bfield = 6.7, Q = 1/mpf(4), area = 0.1):
    phase = 2*mp.pi * bfield * area * 10**(-12) / (2*mp.pi*HBAR /(Q * ELEC))
    def f(x):
        return fexp(1j*phase*x+ 1j*3*mp.pi / 7)
    
    return np.complex128(f(np.linspace(0,.1, PlotPoints)))

    
  

def plotSpectrumDistance((A,B), display_frequencies = False):
    """
    Computes and plots the FFT of a modulating function. 
    """
    V = A.input_parameters["x"][1]
    R = B.rrfunction[-1,:].ravel()
    R = R - R.mean()
    d = 0
    assert R.shape == V.shape
    
    
    M1 = int(R.size * d)
    y, x = np.float64(R[M1:]), np.float64(V[M1:])
    capT = x[-1] - x[0]
    N = x.size
    FSample = N / capT
    
    ####
    deltaV = 1/FSample
    delta_V = deltaV * 10**6
    deltaF = 1/capT
    maxV = x[-1]*10**6
    cap_t = capT*10**6
    f_sample = FSample * 10**(-6)
    dx = (x[1]-x[0]) * 10**6
    print "Total Voltage Range: %.1f [muV]" % cap_t
    print "MaxVoltage: %.1f [muV]" % maxV
    print "Number of samples: %d" % N
    print "Sample rate: %.1f * 10^6" % f_sample
    print "Stepsize: %.5f [muV]" % delta_V
    print "Stepsize 2: %.5f [muV]" % dx
    print "Frequency stepsize: %.1f [1/V]" % deltaF
    ####
    
    
    
    Y = rfft(y*np.hamming(y.size))[1:-1] 
    frq = np.fft.fftfreq(x.size, d=x[1]-x[0])[1:Y.size-1]*10**(-6)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x,y,color='black',linewidth = 1.5)
    ax1.grid(True)
    ax1.set_ybound([-.25, .5])
    ax1.set_yticks([-0.25, 0, .25, .5])
    ax1.set_yticklabels([-0.25, 0, 0.25, 0.5])
    xt = np.linspace(x[0], x[-1], 6)
    xt_labels = [str(i) for i in np.round(xt*10**6, decimals = 1)]    
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt_labels)    
    ax1.set_xlabel(r'Size of one edge [$\mu m$]')
    ax1.set_title(r'Modulating function $(T = 0$ [K], $V=60$ [$\mu$V])')
    ax1.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax1.xaxis.set_label_coords(0.95, -0.1)
    
    ax = fig.add_subplot(212)
    ax.set_title(r'Fourier transform')
    if display_frequencies:
        freqs, labels, lstyle  = DISTfrequencies(A)
        print freqs, labels, lstyle
        for i, j, k in zip(freqs,labels,lstyle):
            plt.axvline(i * 10**(-6), color = 'black', linewidth = 2, 
                        label = j,linestyle = k)
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],'o',linewidth=1,
            color='black',markersize=3) # plotting the spectrum
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],color='black',
            linewidth=.75) 
    ax.set_xlabel(r'Frequency [1/$\mu m$]')
    xt = np.linspace(0, 15, 6)
    ax.set_xticks(xt)
    #ax.set_xticklabels([0,0.05,0.1,0.15,0.2,0.25])        
    ax.set_yticks([0,2,4])
    ax.set_yticklabels(['0','2','4'])
    ax.set_ylabel(r'Amplitude')
    
    ax.legend(loc='upper right', prop={'size':14})
    ax.xaxis.set_label_coords(1.0, -0.1)
  
def DISTfrequencies(A):
    """ Computes the frequencies and corresponding labels for a particle. 
    Input is:
    A = baseparameter object"""
    V = A.V.ravel()
    v = A.input_parameters["v"][0:-1:2]
    Q = A.Q
    f  =np.abs(np.complex128(Q*ELEC*V/(v*2*mp.pi*HBAR)))
    f.sort()
    if v.size == 2:
        labels =  [r'$QeV/(v_ch$)',
                   r'$QeV/(v_nh$)']
        linestyle = ['--',':']
    elif v.size == 1:
        labels =  [r'$f_1$ = $QeV/(vh$)']
        linestyle = ['-','--']
    else:
        labels = ['']*v.size // 2
        
    return [f, labels,linestyle]
    
def plot_AB(B, bfield = 6.6):
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
    
