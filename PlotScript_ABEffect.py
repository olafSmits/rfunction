# -*- coding: utf-8 -*-
"""
Created on Fri Mar 08 14:03:19 2013
"""
from __future__ import division
import sympy.mpmath as mp
from sympy.mpmath import mpf
import numpy as np
import matplotlib.pyplot as plt

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
#   AB OSCILLATIONS --- WEAK / STRONG
#===============================================================================


def plotDistanceEffectOnAB(abelian = False, saving = False):
    A,B, dis = distanceChangeParticleForAB(strong_effect = True,\
                abelian = abelian)
    plot_AB((A,B), ABOscillations(dis.size), dis, saving = saving,
            middle_title_extra = r' (Strong)')
    C,D, dis = distanceChangeParticleForAB(strong_effect = False,\
                abelian = abelian)
    plot_AB((C, D), ABOscillations(dis.size), dis, saving = saving, 
            middle_title_extra = r' (Weak)')    
    return (A,B,dis), (C, D, dis)

def distanceChangeParticleForAB(strong_effect = True, abelian = False):
    """
    Computes modulating function as a function of changing the distance along
    one edge.
    
    Warning: this function can take very long to complete.
    """
    assert isinstance(strong_effect, bool)
    Vpoints = np.array([mpf(50)/mpf(10**6)])
    Nterms= 201
    dist1 = np.array([mpf('2.5')/ mpf(10**(6))]*Nterms)
    
    if strong_effect:
        dist2 = np.linspace(mpf('2.0')/mpf(10**(6)), mpf('3.0')/mpf(10**(6)), Nterms)
    else:
        dist2 = np.linspace(mpf('2.25')/mpf(10**(6)), mpf('2.75')/mpf(10**(6)), 
                            Nterms)
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(9,2),(9,2),(6,2),(6,2)]],
        "c":[1,1,1,1],
        "g":[mpf(1)/mpf(10),mpf(1)/mpf(10),mpf(1)/mpf(8),mpf(1)/mpf(8)],
        "x":[-dist1, dist2, -dist1, dist2]}
    Qe = 1/mpf(4)

    if not strong_effect:
        genData["v"] = [mpf(i) * mpf(10**j) for (i,j) in [(8,3),(8,3),(3,3),(3,3)]]
    
    # Abelian case is not used in text    
    if abelian:
        Qe = 1/mpf(2)
        genData["c"] = [1,1]
        genData["g"] = [1/mpf(2), 1/mpf(2)]
        genData["x"] = [-dist1, dist2]
        if strong_effect:
            genData["v"] = [mpf(i) * mpf(10**j) for (i,j) in [(5,2),(5,2)]]
        else:
            genData["v"] = [mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3)]]
            
    mp.mp.dps= 150
    A = base_parameters(genData, V = Vpoints, Q = Qe, T = 0)
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 500, maxA = 20, maxK = 20)
    B.genAnswer()
    
    return A,B, dist2

    
def ABOscillations(PlotPoints = 40, bfield = 6.7, Q = 1/mpf(4), 
                   max_area_change_percentage = 0.1, total_area = 0.1,
                    give_freq = False, saving = False):
    """

    Estimate the AB oscillations for a given b-field, total area and areachange.
    Assume the total area decreases as:
    [ total_area ] ---> [ total_area ] * [ 1 - max_area_change_percentage ]
    
    and total_area is given in micrometers^2    
    
    The phase is given by: 2*pi*B*S / (h/Qe) + some random component
    and we assume the change in area is linear.
    """
    phase = 2*mp.pi * bfield * total_area * 10**(-12) / (2*mp.pi*HBAR/(Q*ELEC))
    def f(x):
        return fexp(1j*phase*x +1j*3*mp.pi / 7)
    ans = np.complex128(f(np.linspace(0,max_area_change_percentage, 
                                       PlotPoints)))
    if not give_freq:
        return ans
    else:
        return ans, phase/(2*mp.pi)
                                       
    
def plot_AB((A,B), AB_Oscillations, distance, saving= False, middle_title_extra = r''):

    # Sanity checks and conversions
    rfunction_cmplx = np.transpose(B.rfunction).ravel()
    assert rfunction_cmplx.size == distance.size
    assert rfunction_cmplx.size == AB_Oscillations.size
    total_phase = np.real(np.complex128(AB_Oscillations * rfunction_cmplx))
    rfunction_real = np.real(np.complex128(rfunction_cmplx))

    # Normalize
    total_phase = total_phase / np.max(np.abs(AB_Oscillations))        
    AB_Osc_normalized = np.real(AB_Oscillations) / np.max(np.abs(AB_Oscillations))
    
    # Get ticks for x-axis
    xt = np.linspace(distance[0], distance[-1], 7)
    xt = np.float64(xt)
    xt_labels = [str(np.around(i * 10**6, decimals=1)) for i in xt]
    for i, j in enumerate(xt_labels):
        if j[-2:] == '.0': xt_labels[i] = j[:-2]
    
    
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.3)
    
    ########### Plot AB phase ####################
    ax1 =  fig.add_subplot(311)
    ax1.plot(distance, AB_Osc_normalized, linewidth=1.5)
    
    ax1.get_lines()[0].set_color('black')
    ax1.set_title(r'AB oscillations without dynamical interference (normalized)', fontsize=16)
    ax1.set_ylabel(r"$\mathrm{Re}[e^{i\Phi_{AB}}]{}$", fontsize=16)
    ax1.set_ybound([-1,1])
    ax1.set_xbound([np.float64(distance[0]),np.float64(distance[-1])])
    ax1.set_yticks([-1,-.5, 0, .5, 1])
    ax1.set_yticklabels([-1,-0.5, 0, 0.5, 1])
    ax1.set_xticks(xt)
    ax1.set_xticklabels([])
    ax1.grid(True)
    
    ########### Plot Hmod ########################
    ax2 =  fig.add_subplot(312)
    
    ax2.plot(distance, rfunction_real, color = 'black', linewidth=1.5)
    title_middle = r'Modulating function' + middle_title_extra
    ax2.set_title(title_middle, fontsize=16)
    ax2.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax2.set_ybound([-.2,.2])
    ax2.set_xbound([np.float64(distance[0]),np.float64(distance[-1])])
    ax2.set_yticks([-.2,-.1, 0, .1, .2])
    ax2.set_yticklabels([-0.2,-0.1, 0, 0.1, 0.2])
    ax2.set_xticks(xt)
    ax2.set_xticklabels([])
    ax2.grid(True)
    
    ########### Plot total interference ##########
    ax3 =  fig.add_subplot(313)

    ax3.plot(distance, total_phase, linewidth=1.5, color = 'black')
    ax3.set_title(r'Combined interference signal (normalized)', fontsize=16)
    ax3.set_ylabel(r"$\mathrm{Re}[e^{i\Phi_{AB}}H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    if np.max(np.abs(total_phase)) < .1:    
        ax3.set_ybound([-.1,.1])
        ax3.set_yticks([-0.1, -0.05,0,0.05, 0.1])
        ax3.set_yticklabels([-0.1, -0.05,0,0.05, 0.1])    
    elif np.max(np.abs(total_phase)) < .15:    
        ax3.set_ybound([-.15,.15])
        ax3.set_yticks([-0.15, -0.075,0,0.075, 0.15])
        ax3.set_yticklabels([-0.15, -0.075,0,0.075, 0.15])    
    else:
        ax3.set_ybound([-.2,.2])
        ax3.set_yticks([-0.2, -0.1,0,0.1, 0.2])
        ax3.set_yticklabels([-0.2, -0.1,0,0.1, 0.2])       
    ax3.set_xbound([np.float64(distance[0]),np.float64(distance[-1])])
    ax3.set_xticks(xt)
    ax3.set_xticklabels(xt_labels)         
    ax3.grid(True)
    ax3.set_xlabel(r'Length of one edge [$\mu m$]', fontsize=16)
    ax3.xaxis.set_label_coords(.82, -0.2)
    
    
    plt.setp(ax1.get_yticklabels(), fontsize=14)
    plt.setp(ax2.get_yticklabels(), fontsize=14)
    plt.setp(ax3.get_yticklabels(), fontsize=14)
    plt.setp(ax1.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax3.get_xticklabels(), fontsize=14)
    if saving: 
        if middle_title_extra ==  r' (Strong)':
            name = ' signal_edge_variation_opt.pdf'
        elif middle_title_extra ==  r' (Weak)':
            name = ' signal_edge_variation_pess.pdf'
        else:
            name = ' signal_edge_variation.pdf'
        plt.savefig( name, bbox_inches=0, dpi= 300)
    plt.show()    