# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:20:13 2012
"""
import sympy.mpmath as mp
from sympy.mpmath import mpf
import time as time
import numpy as np
import matplotlib.pyplot as plt
import copy

import os
mp.mp.dps= 20
mp.mp.pretty = True

from Interface import Rfunc_constructor, Current
from InputParameters import base_parameters 

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 16})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
_mp = np.vectorize(mp.mpf)

#===============================================================================
# MAIN PLOT (Current + Hmod vs Voltage)
#===============================================================================

def main_plot(saving = False):
    """Main plot:
    Plots the mod-H function, the current with and without interference
    # and transmission"""
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = mpf('2.0')/ mpf(10**(6))
    dist2 = mpf('1.8')/mpf(10**(6))
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(7,3),(7,3),(1,3),(1,3)]],
        "c":[1,1,1,1],
        "g":[mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(1)/mpf(8)],
         "x":[-dist1, dist2, -dist1, dist2]}
    mp.mp.dps= 60    
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = 1/mpf(10**4))
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 450, maxA = 11, maxK = 11)
    single, interference = Current(B)
    
    fig = plt.figure()
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]      
    
    ax = fig.add_subplot(211)  
    ax.plot(Vpoints,interference/np.max(single), label = r"With interference", linewidth=1.5)
    ax.plot(Vpoints,single/np.max(single), label = r"Without interference", linewidth=1.5)                        
    
    ax.get_lines()[1].set_dashes([5,2])    
    for i in ax.get_lines(): i.set_color('black')
  
    ax.set_title(r'Tunnelling current for the Pfaffian state', fontsize=16)
    ax.set_ylabel(r"$I_B/\mathrm{max}(I_B){}$", fontsize=16)
    ax.set_ybound([0,1])
    ax.set_yticks([.25, .5, .75, 1])
    ax.set_yticklabels([0.25, 0.5, 0.75, 1])
    ax.set_xticks(xt)
    ax.set_xticklabels([])
    ax.legend(loc = 'upper right', prop={'size':14})
    plt.setp(ax.get_xticklabels(), fontsize = 14)
    plt.setp(ax.get_yticklabels(), fontsize = 14)    
    ax.grid(True)
    ax2 = fig.add_subplot(212)
    ax2.plot(Vpoints, B.rrfunction, linewidth=1.5) 
    ax2.get_lines()[0].set_color('black')
    ax2.set_title(r'Modulating function for the Pfaffian state', fontsize=16)
    ax2.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax2.set_xlabel(r'Voltage [$\mu$V]', fontsize=16)
    ax2.xaxis.set_label_coords(.9, -0.12)
    ax2.set_yticks([-0.25,0,.25,.5,.75,1])
    ax2.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax2.set_xticks(xt) 
    ax2.set_xticklabels(xt_labels)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), fontsize=14)
    plt.setp(ax2.get_yticklabels(), fontsize=14)        
    if saving: plt.savefig('main_plot.pdf', bbox_inches=0, dpi=300)
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    return B
    
#===============================================================================
# TEMPERATURE VARIATION (Hmod vs Voltage)
#===============================================================================
    
def temperature(saving = False):
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = np.array(mpf('3.5')/ mpf(10**(6)))
    dist2 = np.array(mpf('3.5')/mpf(10**(6)))
    genData = { 
         "v":[mpf(i) * mpf(10**j) for (i,j) in [(9,4),(9,4),(9,3),(9,3)]],
         "c":[1,1,1,1],
         "g":[1/mpf(8), 1/mpf(8), 1/mpf(8), 1/mpf(8)],
         "x":[-dist1, dist2, -dist1, dist2]}
    names = [0,10,18]   
    temperatureset = [mpf(i)/mpf(10**3) for i in names]  
    ans = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, j in enumerate(temperatureset):
        A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = j)
        
        

        if j < 9/mpf(10**3):
            mp.mp.dps = 30
            B = Rfunc_constructor(A, method = 'series')
            B.setParameter(nterms = 500, maxA = 15, maxK = 15)
        else:
            mp.mp.dps = 20
            B = Rfunc_constructor(A, method = 'fortran')
            B.setParameter(nterms = 2*132000, maxA = 25, maxK = 25)
        B.genAnswer()

        ans.append(B)
        ax.plot(Vpoints, B.rrfunction, \
                            label = str(names[i]) +' [mK]', linewidth=1.5) 
                            
    dashstyle  = [(None, None), [7,2], [3,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i])
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]        
    ax.set_title(r'The modulating function \\ for different temperature scales', fontsize=16)
    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax.set_xlabel(r'Voltage [$\mu$V]', fontsize=16)
    ax.xaxis.set_label_coords(.9, -0.06)
    ax.set_ybound([-.25,1.125])
    ax.set_yticks([-0.25,0,.25,.5,.75,1])
    ax.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.grid(True)
    ax.legend(loc='upper right', prop={'size':14})
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    if saving: plt.savefig('temperature.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return ans


##===============================================================================
## PLOTS FOR 5/2 STATE
##===============================================================================


def particles_5_2(saving = False):
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = mpf('2.4')/ mpf(10**(6)) 
    dist2 = mpf('2.1')/mpf(10**(6)) 
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3),(1.4,3),(1.4,3)]],
         "x":[-dist1, dist2, -dist1, dist2]}
    pfaffE2 = {
        "g":[mpf(1)/mpf(2),mpf(1)/mpf(2)],
        "c":[1,1],
        "x":[-dist1, dist2],
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3)]],"Q":1/mpf(2)}
    pfaff = {
        "g":[mpf(1)/mpf(8), mpf(1)/mpf(8),mpf(1)/mpf(8), mpf(1)/mpf(8)], 
        "c":[1,1,1,1],"Q":1/mpf(4)}
    apf = {
        "g":[mpf(1)/mpf(8),mpf(1)/mpf(8),mpf(3)/mpf(8), mpf(3)/mpf(8)], 
        "c":[1,1,-1,-1],"Q":1/mpf(4)}
    state331 = {
        "g":[mpf(1)/mpf(8), mpf(1)/mpf(8), mpf(1)/mpf(4), mpf(1)/mpf(4)], 
        "c":[1,1,1,1],"Q":1/mpf(4)}

    particleset = [apf, state331, pfaff, pfaffE2]
    names = [
            "Anti-Pfaffian (e/4)",                
            "(3,3,1)-state (e/4)",
            "Pfaffian (e/4)", 
            "Laughlin (e/2)"]

    mp.mp.dps= 60
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ans = []
    for i in range(0,4):
        particle = genData.copy()
        particle.update(particleset[i])
        Qe = particle["Q"]
        del particle["Q"]
        A = base_parameters(particle, V = Vpoints, Q = Qe, T = 1/mpf(10**3))
        B = Rfunc_constructor(A, method = 'series')
        B.setParameter(nterms = 1200, maxA = 12, maxK = 12)
        B.genAnswer()
###        uncomment to return all instances
#        ans.extend([copy.deepcopy(A),copy.deepcopy(B)])
###
        if B.rrfunction.shape[1] > 1:
            ax.plot(Vpoints, B.rrfunction[:,0], label = names[i], linewidth=1.5) 
        else:
            ax.plot(Vpoints, B.rrfunction, label = names[i], linewidth=1.5)             

        
    dashstyle  = [(None, None), [10,4], [5,3,1,3], [2,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i])            
        
    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax.set_xlabel(r'Voltage [$\mu$V]', fontsize=16)
    ax.xaxis.set_label_coords(.9, -0.06)
    ax.set_ybound([-.5, 1])
    ax.set_yticks([-0.25, 0, .25, .5, .75, 1])
    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.set_title(r"Modulating function for $\nu = 5/2 $ candidates", fontsize=16)
    ax.legend(loc='upper right', prop={'size':14})
    ax.grid(True)    
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    if saving: plt.savefig('particles_5_2.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return ans


##===============================================================================
## PLOTS FOR 7/3 STATE
##===============================================================================


def particles_7_3(saving = False):
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = mpf('2.4')/ mpf(10**(6)) 
    dist2 = mpf('2.1')/mpf(10**(6)) 
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3),(1.4,3),(1.4,3)]],
         "x":[-dist1, dist2, -dist1, dist2]}
    LaughlinE3 = {
        "g":[mpf(1)/mpf(2),mpf(1)/mpf(2)],
        "c":[1,1],
        "x":[-dist1, dist2],
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3)]],"Q":1/mpf(3)}
    aBS23 = {
        "g":[mpf(1)/mpf(3), mpf(1)/mpf(3),mpf(5)/mpf(8), mpf(5)/mpf(8)], 
        "c":[1,1,-1,-1],"Q":1/mpf(3)}
    BS13 = {
        "g":[mpf(1)/mpf(3),mpf(1)/mpf(3),mpf(3)/mpf(8), mpf(3)/mpf(8)], 
        "c":[1,1,1,1],"Q":1/mpf(3)}
    aRR4 = {
        "g":[mpf(1)/mpf(12), mpf(1)/mpf(12), mpf(1)/mpf(4), mpf(1)/mpf(4)], 
        "c":[1,1,-1,-1],"Q":1/mpf(6)}


    particleset = [aBS23, BS13, aRR4, LaughlinE3]
    names =[    r"$\overline{\mathrm{BS}}_{2/3}$ (e/3)",
                r"$\mathrm{BS}_{1/3}^{\psi}$ (e/3)",
                r"$\overline{\mathrm{RR}}_{k=4} $ (e/6)",
                r"Laughlin (e/3)"]
    mp.mp.dps= 60    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ans = []
    for i in range(len(names)):
        particle = genData.copy()
        particle.update(particleset[i])
        Qe = particle["Q"]
        del particle["Q"]
        A = base_parameters(particle, V = Vpoints, Q = Qe, T = 1/mpf(10**3))
        
        B = Rfunc_constructor(A, method = 'series')
        B.setParameter(nterms = 1200, maxA = 12, maxK = 12)
        B.genAnswer()
        #ans.append(B)
        ax.plot(Vpoints, B.rrfunction, label = names[i], linewidth=1.5) 

    dashstyle  = [(None, None), [10,4], [5,3,1,3], [2,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i]) 

    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax.set_xlabel(r'Voltage [$\mu$V]', fontsize=16)
    ax.xaxis.set_label_coords(.9, -0.06)
    ax.set_ybound([-.5, 1])
    ax.set_yticks([-0.25, 0, .25, .5, .75, 1])
    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]    
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.set_title(r"Modulating function for $\nu = 7/3 $ candidates", fontsize=16)
    ax.legend(loc='upper right', prop={'size':14})
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    if saving: plt.savefig('particles_7_3.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return ans
    

def particles_12_5(saving = False):
    Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201)
    dist1 = mpf('2.4')/ mpf(10**(6)) 
    dist2 = mpf('2.1')/mpf(10**(6)) 
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3),(1.4,3),(1.4,3)]],
         "x":[-dist1, dist2, -dist1, dist2]}
    LaughlinE5 = {
        "g":[mpf(2)/mpf(5),mpf(2)/mpf(5)],
        "c":[1,1],
        "x":[-dist1, dist2],
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(5,3),(5,3)]],"Q":1/mpf(3)}
    BS25 = {
        "g":[mpf(1)/mpf(10),mpf(1)/mpf(10),mpf(1)/mpf(8), mpf(1)/mpf(8)], 
        "c":[1,1,1,1],"Q":1/mpf(5)}
    HH25 = {
        "g":[mpf(1)/mpf(5),mpf(1)/mpf(5),mpf(2)/mpf(5), mpf(2)/mpf(5)], 
        "c":[1,1,1,1],"Q":1/mpf(5)} 
    aRR3 = {
        "g":[mpf(1)/mpf(10), mpf(1)/mpf(10), mpf(3)/mpf(10), mpf(3)/mpf(10)], 
        "c":[1,1,-1,-1],"Q":mpf(1)/mpf(5)}


    particleset = [HH25,aRR3, BS25, LaughlinE5]
    names =[    r"$\mathrm{HH}_{2/5}$ (e/5)",
                r"$\overline{\mathrm{RR}}_{k=3} $ (e/5)",
                r"$\mathrm{BS}_{2/5}$ (e/5)",
                r"Laughlin (2e/5)"]
    mp.mp.dps= 60    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ans = []
    for i in range(len(names)):
        particle = genData.copy()
        particle.update(particleset[i])
        Qe = particle["Q"]
        del particle["Q"]
        A = base_parameters(particle, V = Vpoints, Q = Qe, T = 1/mpf(10**3))
        
        B = Rfunc_constructor(A, method = 'series')
        B.setParameter(nterms = 1000, maxA = 12, maxK = 12)
        B.genAnswer()
        ans.append(B)
        ax.plot(Vpoints, B.rrfunction, label = names[i], linewidth=1.5) 
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]
    dashstyle  = [(None, None), [10,4], [6,3,1,3,1,3],[2,2] ,[2,6]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i]) 

    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax.set_xlabel(r'Voltage [$\mu$V]', fontsize=16)
    ax.xaxis.set_label_coords(.9, -0.06)
    ax.set_ybound([-.5, 1])
    ax.set_yticks([-0.25, 0, .25, .5, .75, 1])
    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]    
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.set_title(r"Modulating function for $\nu = 12/5 $ candidates", fontsize=16)
    ax.legend(loc='upper right', prop={'size':14})
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    if saving: plt.savefig('particles_12_5.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return ans    
#def plot_all(saveon = False):
#    os.chdir('C:\\Science\\LaTeX\\Interferometer PRB')
#    main_plot(saving = saveon)
#    temperature(saving = saveon)
#    particles_5_2(saving = saveon)
#    particles_7_3(saving = saveon)
##A = RCNTC.RfuncCNTC(par = genData, nterms = 100000)
##A.updateParameters(par =  {
##    "v":[mpf('5.')* mpf(10**(4)),mpf('5.')*mpf(10**(3))],\
##    "g":[1/mpf(8),1/mpf(8)],\
##    "c":[1,1],\
##    "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.3')/mpf(10**(6)), \
##    "T" :mpf(8) / mpf(10**(3)), "Q" :1/mpf(4)})
##A.updateParameters(par = pfaffData)
##A.genAnswer()


#===============================================================================
# MULTIPLE MODES -- NOT YET IMPLEMENTED
#===============================================================================


def multi_modes(saving = False):
    """Plots the modulating function and interference current for an edge with
    3 modes."""
    Vpoints = mp.linspace(0, mpf('2.')/mpf(10**4), 201)
    dist1 = np.array(mpf('4.5')/ mpf(10**(6)))
    dist2 = np.array(mpf('3.')/mpf(10**(6)))
    genData = { 
        "v":[mpf(i) * mpf(10**j) for (i,j) in [(10,3),(10,3),(3,3),(3,3),(5,3),(5,3)]],
        "x":[dist1, -dist2, dist1, -dist2,dist1, -dist2],
        "g":[mpf(1)/mpf(10), mpf(1)/mpf(10), mpf(1)/mpf(10), mpf(1)/mpf(10),mpf(1)/mpf(10), mpf(1)/mpf(10)], 
        "c":[1,1,1,1,1,1]}        

                            
    mp.mp.dps= 90    
    A = base_parameters(genData, V = Vpoints, Q = 1/mpf(4), T = mpf(0))
    B = Rfunc_constructor(A, method = 'series')
    B.setParameter(nterms = 800)
    single, interference = Current(B)
    
    fig = plt.figure()
    xt = np.linspace(0, 2 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]      
    
    ax = fig.add_subplot(211)  
    ax.plot(Vpoints,interference/np.max(single), label = r"With interference", linewidth=1.5)
    ax.plot(Vpoints,single/np.max(single), label = r"With interference", linewidth=1.5)                        
    
    ax.get_lines()[1].set_dashes([5,2])    
    for i in ax.get_lines(): i.set_color('black')
  
    ax.set_title(r'Tunnelling current for edge with three modes')
    ax.set_ylabel(r"$I_B/\mathrm{max}(I_B){}$")
    ax.set_ybound([0,1])
    ax.set_yticks([0,.25, .5, .75, 1])
    ax.set_yticklabels([0,0.25, 0.5, 0.75, 1])
    ax.set_xticks(xt)
    ax.set_xticklabels([])
    ax.legend(loc = 'upper right', prop={'size':14})
    plt.setp(ax.get_xticklabels(), fontsize = 14)
    plt.setp(ax.get_yticklabels(), fontsize = 14)    
    ax.grid(True)
    ax2 = fig.add_subplot(212)
    ax2.plot(Vpoints, B.rrfunction, linewidth=1.5) 
    ax2.get_lines()[0].set_color('black')
    ax2.set_title(r'Modulating function for edge with three modes', fontsize=16)
    ax2.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$", fontsize=16)
    ax2.set_xlabel(r'Volt [$\mu$V]')
    ax2.set_yticks([-0.25,0,.25,.5,.75,1])
    ax2.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax2.set_xticks(xt) 
    ax2.set_xticklabels(xt_labels)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), fontsize=12.)
    plt.setp(ax2.get_yticklabels(), fontsize=12.)        
    if saving: plt.savefig('main_plot.pdf', bbox_inches=0, dpi=300)
    plt.show()
    return B

if __name__ == '__main__':
    pass