# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:29:21 2013
"""

from __future__ import division
import sympy.mpmath as mp
from sympy.mpmath import mpf
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft

mp.mp.dps= 20
mp.mp.pretty = True

from Interface import Rfunc_constructor
from InputParameters import base_parameters, ELEC, HBAR

_mp = np.vectorize(mp.mpf)
fcos = np.vectorize(mp.cos)
fexp = np.vectorize(mp.exp)
plt.rc('text', usetex=False)

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 14})
plt.rc('text', usetex=True)


#===============================================================================
# FREQUENCY ANALYSIS -- CHANGING DISTANCE ALONG ONE EDGE    
#===============================================================================

def plotFrequencySpectrumWhenChangingDistance(saving = False):
    """
    Script that comptues and plots it all at once. Warning: computing the
    R function takes a long time.
    """
    A,B  = distanceChangeParticle()
    plotSpectrumDistance((A,B), display_frequencies = True, saving = saving)
    return A,B

def distanceChangeParticle():
    """
    Computes modulating function as a function of changing the distance along
    one edge.
    
    Warning: this function can take very long to complete.
    """
    Vpoints = np.array([mpf(60)/mpf(10**6)])
    Nterms= 201
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    dist2 = np.linspace(mpf('0.001')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), Nterms)
    
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(1,3),(1,3),(6,2),(6,2)]],
        "c":[1,1,1,1],
        "g":[mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8)],
        "x":[-dist1, dist2, -dist1, dist2]}
    mp.mp.dps= 150
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 700)
    B.genAnswer()
    
    return A,B


    
def DISTfrequencies(A):
    """ Computes the frequencies and createscorresponding labels for a particle.
    The frequencies correspond to the oscillations found in the H-mod function
    when the voltage is varied.
    Input is:

    A = baseparameter object
    
    Frequencies are given by: exp(Qe x/(v*h)) for all combinations of v and x
    (velocities and distances)"""
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
    return [f, labels, linestyle] 

def plotSpectrumDistance((A,B), display_frequencies = True, saving = False):
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
  
    Y = rfft(y*np.hamming(y.size))[1:-1] 
    frq = np.fft.fftfreq(x.size, d=x[1]-x[0])[1:Y.size-1]*10**(-6)

    ###### Plot the H-mod ######
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.4)
    
    ax1 = fig.add_subplot(211)
    ax1.plot(x,y,color='black',linewidth = 1.5)
    ax1.grid(True)

    ax1.set_title(r'Modulating function $(T = 0, V=60 ~[\mu\mathrm{V}]){}$', fontsize=16)

    xt = np.linspace(x[0], x[-1], 6)
    xt_labels = [str(i) for i in np.round(xt*10**6, decimals = 1)]     
    xt_labels[0] ='0'    
    xt_labels[-1] = '3'
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt_labels)    
    ax1.set_xlabel(r'Length of one edge [$\mu m$]', fontsize=16)
    ax1.xaxis.set_label_coords(0.82, -0.13)
    
    ax1.set_ybound([-.25, .25])
    ax1.set_yticks([-0.25,-.125, 0,.125, .25])
    ax1.set_yticklabels([-0.25, -0.125, 0,0.125, 0.25])
    ax1.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)

    
    ###### Plot the FFT and overlay the computed frequencies ######
    ax = fig.add_subplot(212)
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],'o',linewidth=1,
            color='black',markersize=3) # plotting the spectrum
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],color='black',
            linewidth=.75) 
            
    ax.set_title(r'Fourier transform', fontsize=16)
        
    xt = np.linspace(0, 10, 6)
    xlabels = [str(int(x)) for x in xt]
    ax.set_xbound([0, 11])    
    ax.set_xticks(xt)
    ax.set_xticklabels(xlabels)       
    ax.set_xlabel(r'Frequency [1/$\mu m$]', fontsize=16)
    ax.xaxis.set_label_coords(0.87, -0.13)
        
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['0','1','2','3','4'])    
    ax.set_ylabel(r'Amplitude', fontsize=16)
    
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_xticklabels(), fontsize=14)

    # Compute frequencies and display in plot:

    if display_frequencies:
        freqs, labels, lstyle  = DISTfrequencies(A)
        for i, j, k in zip(freqs,labels,lstyle):
            plt.axvline(i * 10**(-6), color = 'black', linewidth = 2, 
                        label = j,linestyle = k)
                        
        ax.legend(loc='upper right', prop={'size':14})  
    if saving: plt.savefig('ft_analysis_edge_variation.png', bbox_inches=0, dpi=fig.dpi)
    plt.show
  