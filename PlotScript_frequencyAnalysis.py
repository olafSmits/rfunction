# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:45:00 2013
"""

from __future__ import division
import sympy.mpmath as mp
from sympy.mpmath import mpf

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft

mp.mp.dps= 20
mp.mp.pretty = True

from Interface import Rfunc_constructor, Current
from InputParameters import base_parameters, ELEC, HBAR


plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 14})
plt.rc('text', usetex=True)


#===============================================================================
# PLOTS FOR FREQUENCY ANALYSIS    
#===============================================================================

def plotFrequency(saving = False):
    """
    This little script glues it all together: it computes the H-mod function,
    and plots the spectrum.    
    """
    particle = frequencyParticle(plotting = False)
    SpectrumPlot(particle, display_frequencies = True, saving = saving)
    return particle

def frequencyParticle(plotting = True):
    """This function constructs and computes the modulating functions,
    which will be used in the frequency analysis
    
    It comes with an optional plotting instruction -- this was used to look for
    a "nice" parameter set.    
    """
    # Voltages array and distances.
    Vpoints = mp.linspace(0, mpf('2.5')/mpf(10**4), 301)
    dist1 = np.array(mpf('2.4')/ mpf(10**(6)))
    dist2 = np.array(mpf('1.1')/mpf(10**(6)))
    
    # We did not have a particular anyon in mind -- frequency is
    # independent of the g-values anyway.
    particletype = { \
         "g":[mpf(1)/mpf(8), mpf(1)/mpf(8), mpf(1)/mpf(6), mpf(1)/mpf(6)], 
         "c":[1,1,1,1], 
         "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3),(1,3),(1,3)]],
         "x":[-dist1, dist2, -dist1, dist2],
         "Q":1/mpf(4)}
        
            
    particleset = [particletype]
    if plotting:
        # plotting is optional
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    # we can add more particle types to  particleset, and loop over these        
    for particle in particleset:
        Qe = particle["Q"]
        del particle["Q"]

        # because of the large range of the voltage and low values of the 
        # velocities we enable many oscillations in the spectrum
        # The downside is that we require an insane precision (170 digits)
        mp.mp.dps= 170

        # The T=0 function behaves requires high precision numbers. 
        # At finite temperatures we run into two problems:
        # (1) Need high precision numbers
        # (2) Need large number of terms
        # We can only fix one of these at a time -- the fortran + cnct method
        # can handle (to some extent) the large number of terms, but does not
        # support arbitrary precision
        
        # Conclusion: stick to T=0 and you only encounter problems with
        # lack of high-precision numbers
        A = base_parameters(particle, V = Vpoints, Q = Qe, T = 0)
        B = Rfunc_constructor(A, method = 'series')
        B.setParameter(nterms = 1000)
        B.genAnswer()
        
        # fix this output if you are looking at multiple particle species
        ans = [A,B]
        if plotting:
            ax.plot(Vpoints, B.rrfunction, linewidth=1.5) 
        particle["Q"] = Qe
        
    # plotting instructions. Dashstyle will give problems for more than 
    # 3 particles -- just add more styles.
    if plotting:
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
        plt.setp(ax.get_xticklabels(), fontsize=12.)
        plt.setp(ax.get_yticklabels(), fontsize=12.)
        plt.show()
    return ans



def computeFrequencies(A):
    """ Computes the frequencies and createscorresponding labels for a particle.
    The frequencies correspond to the oscillations found in the H-mod function
    when the voltage is varied.
    Input is:

    A = baseparameter object
    
    Frequencies are given by: exp(Qe x/(v*h)) for all combinations of v and x
    (velocities and distances)
    
    """
    # Get velocities and distances
    x = A.input_parameters["x"].ravel()
    v = A.input_parameters["v"].ravel()
    assert x.size == v.size
    Q = A.Q
    # Compute frequencies and sort
    f  =np.abs(np.complex128(Q*ELEC*x/(v*2*mp.pi*HBAR)))
    f.sort()
    if v.size == 4:
        labels =  [r'$Qeb/(v_ch$)',
                   r'$Qea/(v_ch$)',
                   r'$Qeb/(v_nh$)',
                   r'$Qea/(v_nh$)']
        linestyle = ['-','--','-.',':']
    elif v.size == 2:
        labels =  [r'$f_1$ = $e^*a/(vh$)',
                   r'$f_1$ = $e^*b/(vh$)']
        linestyle = ['-','--']
    else:
        # Not implemented
        labels = ['']*x.size
        linestyle = ['-']*x.size
        
    return [f, labels,linestyle]
    


def SpectrumPlot((A,B), display_frequencies = True, saving = False):
    """
    Given a base parameter A and a computed H-mod (B) this functions plots
    the function and its Fourier Transform.
    
    The transform is a windowed FFT.
    """
    V = B.V.ravel()
    R = B.rrfunction.ravel()
    d = 0 #  0<=d< 1 --- the beginning [100*d %] is ignored for the FFT.
    assert R.shape == V.shape
    
    
    M1 = int(R.size * d)
    y, x = np.float64(R[M1:]), np.float64(V[M1:])
    capT = x[-1] - x[0]
    N = x.size
    FSample = N / capT
    
    #### Some info on the FFT ####
    deltaV = 1/FSample
    delta_V = deltaV * 10**6
    deltaF = 1/capT
    cap_t = capT*10**6
    f_sample = FSample * 10**(-6)
    dx = (x[1]-x[0]) * 10**6
    print "Total Voltage Range: %.1f [muV]" % cap_t
    print "Number of samples: %d" % N
    print "Sample rate: %.1f * 10^6" % f_sample
    print "Stepsize: %.5f [muV]" % delta_V
    print "Stepsize 2: %.5f [muV]" % dx
    print "Frequency stepsize: %.1f [1/V]" % deltaF
    ####
    
    
    # Windowed FFT (for real input):
    Y = rfft(y*np.hamming(y.size))[1:-1] 
    # Compute corresponding values for frequency
    # selects only the positive frequencies
    frq = np.fft.fftfreq(x.size, d=x[1]-x[0])[1:Y.size-1] * 10**(-6)
    
    ###### Plot the H-mod ######
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x,y,color='black',linewidth = 1.5)
    ax1.grid(True)

    ax1.set_title(r'Modulating function $(T = 0)$', fontsize=16)

    xt = np.linspace(0, 2.5 * 10**(-4), 6)
    xt_labels = [str(int(i * 10**6)) for i in xt]    
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xt_labels)    
    ax1.set_xlabel(r'Voltage [$\mu V$]', fontsize=16)
    ax1.xaxis.set_label_coords(0.9, -0.13)
    
    ax1.set_ybound([-.25, .5])
    ax1.set_yticks([-0.25, 0, .25, .5])
    ax1.set_yticklabels([-0.25, 0, 0.25, 0.5])
    ax1.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)

    
    ###### Plot the FFT and overlay the computed frequencies ######
    ax = fig.add_subplot(212)
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],'o',linewidth=1,
            color='black',markersize=3) # plotting the spectrum
    ax.plot(frq[:frq.size*2//5],abs(Y)[:frq.size*2//5],color='black',
            linewidth=.75) 

    ax.set_title(r'Fourier transform', fontsize=16)
    
    xt = np.linspace(0, 0.25, 6)
    ax.set_xticks(xt)
    ax.set_xticklabels([0,0.05,0.1,0.15,0.2,0.25])        
    ax.set_xlabel(r'Frequency [1/$\mu$V]', fontsize=16)
    ax.xaxis.set_label_coords(0.85, -0.13)
        
    ax.set_yticks([0,2,4])
    ax.set_yticklabels(['0','2','4'])    
    ax.set_ylabel(r'Amplitude', fontsize=16)
    
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax1.get_xticklabels(), fontsize=14)

    # Compute frequencies and display in plot:
    if display_frequencies:
        freqs, labels, lstyle  = computeFrequencies(A)
        for i, j, k in zip(freqs,labels,lstyle):
            plt.axvline(i * 10**(-6), color = 'black', linewidth = 2, 
                        label = j,linestyle = k)
        ax.legend(loc='upper right', prop={'size':14})
    plt.subplots_adjust(hspace=0.4)
    if saving: plt.savefig('ft_analysis_voltage.pdf', bbox_inches=0, dpi=300)
    plt.show()



    
    
#===============================================================================
# NOT CORRECT -- propagator for three modes
#===============================================================================
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
    if saving: plt.savefig('multi_mode_error.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return B


if __name__ == '__main__':
    pass