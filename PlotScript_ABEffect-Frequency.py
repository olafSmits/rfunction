# -*- coding: utf-8 -*-
"""
Created on Fri Mar 08 14:04:17 2013
"""
from __future__ import division
import sympy.mpmath as mp
from sympy.mpmath import mpf
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from matplotlib.ticker import NullFormatter
from PlotScript_ABEffect import ABOscillations
from PlotScript_edgeVariation import DISTfrequencies

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
# AB OSCILLATION -- FREQUENCY CHANGE
#===============================================================================

def plotInterferenceAndFourier(saving = False):
    
    A, B, dist = generateParticleForABFrequency()
    plotMultiSpectrum((A,B), dist, saving = True)
    return A, B, dist

def generateParticleForABFrequency():
    """
    This function generates the R function for four different voltages, as a 
    function of a changing edge. All other parameters are fixed. The result
    is used for th Fourier Analysis of the AB oscillations at different voltages.
    """
    
    Vpoints = np.array([mpf(i)/mpf(10**6) for i in [10, 20, 30, 40]])
    Nterms= 201
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    dist2 = np.linspace(mpf('2.0')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), Nterms)

    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(9,2),(9,2),(5,2),(5,2)]],
        "c":[1,1,1,1],
        "g":[1/mpf(8), 1/mpf(8), 1/mpf(8), 1/mpf(8)],
        "x":[-dist1, dist2, -dist1, dist2]}

    mp.mp.dps= 150
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 550)
    B.genAnswer()
    
    return A, B, dist2



def plotMultiSpectrum((A,B), dist2, saving = False):
    # Setting up arrays to plot: Interference signal and FFT
    AB, AB_freq = ABOscillations(dist2.size, bfield = 6.8, give_freq = True)
    interferenceSignal = [ np.real(np.complex128(AB*\
                    np.transpose(B.rfunction)[:,j] )) for j in range(0,4)]
    freq_amp = []
    for y in interferenceSignal:                       
        freq_amp.append(rfft(y)[1:-1])
    if np.max(np.abs(freq_amp)) > 10**15:
        freq_amp = [ i * 10**(-14) for i in freq_amp ]
    
    freq_size = freq_amp[0].size
    freq_range = np.fft.fftfreq(dist2.size, d=dist2[1]-dist2[0])[1:freq_size-1] 

    # Predicted frequences for AB, H-mod and combined signal
    Vcopy = A.V.copy()
    freqs, labels, dashstyle = [], [], []
    for i in Vcopy:
        A.V = np.array([i])
        f, l, d = DISTfrequencies(A)
        freqs.append(f)
        labels.append(l)
        dashstyle.append(d)
    A.V = Vcopy
    
    fig = plt.figure()
    
    xticks = np.linspace(dist2[0],dist2[-1], 5)
    xticks_labels = [ str(int(i*10**8) /100) for i in xticks]
    print xticks_labels
    for i in range(4):        
        ax = plt.subplot2grid( (4,2), (i, 0))
        
        
        ax.plot(dist2, interferenceSignal[i])
        ax.set_xticks(xticks)
        if i == 0: ax.set_title(r'Interference signal', fontsize = 16)
        if i == 2:
            ax.set_ylabel(r"$\mathrm{Re}[e^{i\Phi_{AB}}H_{ij}^{\mathrm{mod}}]{}$", fontsize = 16)
        if i == 3: 
            ax.set_xticklabels(xticks_labels)  
            ax.set_xlabel(r'Length of one edge [$\mu m$]', fontsize = 16)
            ax.xaxis.set_label_coords(0.6, -0.22)
        
        else: ax.set_xticklabels([])
        
        ax.set_ybound([-0.2,0.2])
        ax.set_yticks([-0.1,0,0.1])
        ax.set_yticklabels([0.1,0,0.1])
        
        ax.grid(True)
        ax.get_lines()[0].set_color('black')
        ax.get_lines()[0].set_linewidth(1.25)
        
        plt.annotate(str(int(A.V[i]*10**6))+'$[\mu V]$', xy=(0.83, 0.05), 
                     xycoords='axes fraction', fontsize = 12)
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        plt.setp(ax.get_xticklabels(), fontsize = 14)
        plt.setp(ax.get_yticklabels(), fontsize = 14)  


    xticks = np.linspace(2,16, 5)    *10**6
    xticks_labels = [str(int(i)) for i in np.linspace(20,160, 5)]
    yticks = [2,4,6]
    yticks_labels = [2,4,6]
    for i in range(4):        
        ax = plt.subplot2grid( (4,2), (i, 1))
        ax.plot(freq_range[:1 * freq_size//6], 
                np.abs(freq_amp[i])[:1 * freq_size//6],'o',linewidth=1,
                color='black',markersize=3)
        ax.plot(freq_range[:1 * freq_size//6], 
                np.abs(freq_amp[i])[:1 * freq_size//6],linewidth=.75, 
                color='black')
        ax.set_xticks(xticks)
        
        ax.set_ybound([0,int(np.max(np.abs(freq_amp[i]))) + 3])
        plt.axvline(AB_freq * 10**(5), color = 'black', linewidth = 1.5,
                    linestyle ='--', label = '$\phi_{AB}$')
        fv, fc = freqs[i]
        plt.axvline(AB_freq * 10**(5) + fc, color = 'black', linewidth = 1.5,
                    linestyle =':', label = '$\phi_{AB} + f_c$')
        plt.axvline(AB_freq * 10**(5) + fv, color = 'black', linewidth = 1.5,
                    linestyle ='-.', label = '$\phi_{AB} + f_n$')
        if i ==0: 
            ax.legend(prop={'size':14})
            l = plt.legend(bbox_to_anchor=(0, 0, 0.91, 0.915), 
                           bbox_transform=plt.gcf().transFigure,
                            prop={'size':12})
            ax.set_title(r'Fourier Transform', fontsize = 16)
        if i == 2:
            ax.set_ylabel(r'Amplitude', fontsize = 16)
            ax.yaxis.set_label_position("right")
        if i == 3: 
            ax.set_xticklabels(xticks_labels)
            ax.set_xlabel(r'Frequency [1/$\mu m$]', fontsize = 16)
            ax.xaxis.set_label_coords(0.6, -0.22)
        else: ax.set_xticklabels([])
        ax.yaxis.tick_right()        
        if i == 2:
            ax.set_yticks([3,6,9])
            ax.set_yticklabels([3,6,9])
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks_labels)
        ax.get_lines()[0].set_color('black')
        ax.get_lines()[0].set_linewidth(1.25)
        plt.setp(ax.get_xticklabels(), fontsize = 14)
        plt.setp(ax.get_yticklabels(), fontsize = 14)  

    if saving: plt.savefig('interference_ab_fourier_progress.png', bbox_inches=0, dpi=fig.dpi)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()
        
        

#
#
#
#def plotABandModulatingFunctionSubPlot():
#    V = B.V.ravel()
#    R = B.rrfunction.ravel()
#    d = 0 #  0<=d< 1 --- the beginning [100*d %] is ignored for the FFT.
#    
#    assert R.shape == V.shape
#    
#    
#    M1 = int(R.size * d)
#    y, x = np.float64(R[M1:]), np.float64(V[M1:])
#    capT = x[-1] - x[0]
#    N = x.size
#    FSample = N / capT
#    
#    #### Some info on the FFT ####
#    deltaV = 1/FSample
#    delta_V = deltaV * 10**6
#    deltaF = 1/capT
#    cap_t = capT*10**6
#    f_sample = FSample * 10**(-6)
#    dx = (x[1]-x[0]) * 10**6
#    print "Total Voltage Range: %.1f [muV]" % cap_t
#    print "Number of samples: %d" % N
#    print "Sample rate: %.1f * 10^6" % f_sample
#    print "Stepsize: %.5f [muV]" % delta_V
#    print "Stepsize 2: %.5f [muV]" % dx
#    print "Frequency stepsize: %.1f [1/V]" % deltaF
#    ####
#    
#    
#    # Windowed FFT (for real input):
#    Y = rfft(y*np.hamming(y.size))[1:-1] 
#    # Compute corresponding values for frequency
#    # selects only the positive frequencies
#    
#    

