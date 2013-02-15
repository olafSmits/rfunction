# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:39:23 2013
"""
from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
from numpy import newaxis, vectorize
from sympy.mpmath import mpf
from InputParameters import GLOBAL_TEMP, EMP, ELEC, HBAR


b = A[0]

v1 = b.input_parameters["v"][0]
v2 = b.input_parameters["v"][-1]
a1 = (b.input_parameters["x"][0] - b.input_parameters["x"][1])/2



l1 = 2*np.pi/(ELEC/4*(a1)) * abs((1/v1 ))**(-1) * HBAR    
l2 = 2*np.pi/(ELEC/4*(a1)) * abs((1/v2))**(-1) * HBAR    

f1 = 1/l1[0] 
f2 = 1/l2[0]
f1 = ELEC*a1/(v1*4*2*np.pi*HBAR)
f2 = ELEC*a1/(v2*4*2*np.pi*HBAR)
print f1/f2
print (v1+v2)/(v1-v2)


def plotSpectrum(R, V, d):
    """
    N = V.size = # points (equally spaced)
    capT = V[-1] - V[0] = Range of V
    dT = capT / N = 1/Fsample
    dF = frequency resolution =  1/capT = FSample / N
    FFT returns values from -Fsample/2 to Fsample/2
    """
    assert R.shape == V.shape
    M1 = int(R.size) * d
    y, x = np.float64(R[M1:]), np.float64(V[M1:])
    sample_rate = (x[-1] - x[0])/x.size
    
    capT = x[-1] - x[0]
    N = x.size
    FSample = N / capT
    
    ####
    deltaV = 1/FSample
    delta_V = deltaV * 10**6
    deltaF = 1/capT
    print deltaF
    cap_t = capT*10**6
    f_sample = FSample * 10**(-6)
    dx = (x[1]-x[0]) * 10**6
    print "Total Voltage Range: %.1f [muV]" % cap_t
    print "MaxVoltage: %.1f [muV]" % x[-1]
    print "Number of samples: %d" % N
    print "Sample rate: %.1f * 10^6" % f_sample
    print "Stepsize: %.5f [muV]" % delta_V
    print "Stepsize 2: %.5f [muV]" % dx
    print "Frequency stepsize: %.1f [1/V]" % deltaF
    ####
    
    
    
    Y = rfft(y*np.hamming(y.size))[1:-1] 
    frq = np.fft.fftfreq(N, d=x[1]-x[0])[1:Y.size-1] * 10**(-6)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x,y)
    
    ax1.set_ybound([-.5, .5])
    ax1.set_yticks([-0.25, 0, .25, .5])
    ax1.set_yticklabels([-0.25, 0, 0.25, 0.5])
    xt = np.linspace(0,3 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]    
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt_labels)    
    ax1.set_xlabel(r'[$\mu$V]')
    ax1.set_title(r'R function for Pfaffian (T=0 [K])')
    #g1 = np.sin(2*np.pi*np.float64(f1*x)) * .25 + .75
    #g2 = np.sin(2*np.pi*np.float64(f2*x)) * .25 + 1.25
    #ax1.plot(x, g1, color = 'purple')
    #ax1.plot(x, g2, color = 'green')
    
    ax1.xaxis.set_label_coords(1.05, -0.1)
    ax = fig.add_subplot(212)
    ax.set_title(r'Fourier transform')
    plt.axvline(f1 * 10**(-6), color = 'purple', linewidth = 2, label = r'$f_1$ = $e^*x/(v_c\pi\hbar$)')     
    plt.axvline(f2 * 10**(-6), color = 'green', linewidth = 2, label = r'$f_2$ = $e^*x/(v_n\pi\hbar$)')     
    ax.plot(frq[:frq.size//4],abs(Y)[:frq.size//4],'x') # plotting the spectrum
    ax.plot(frq[:frq.size//4],abs(Y)[:frq.size//4],'r') 
    ax.set_xlabel(r'FFT [1/$\mu$V]')
    ax.legend(loc='upper right', prop={'size':14})
    ax.xaxis.set_label_coords(1.05, -0.1)
    return Y, frq    
     
#ff1 = 11.3;   # frequency of the signal
#ff2 = 30.6
#y = sin(2*pi*ff1*t) + sin(2*pi*ff2*t)
#A = A[0]
Dlist = [0.05]
for d in Dlist:
    q = A[1].rrfunction[:,0]
    q = q - q.mean()
    a = plotSpectrum(q, A[1].V, d)
    plt.show()
#
#
#fourier = np.fft.fft(y)
#n = y.size
#timestep = 1.0/Fs
#freq = np.fft.fftfreq(n, d=timestep)
#freq
