# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:20:13 2012
"""
import sympy.mpmath as mp
from sympy.mpmath import mpf
from sympy.mpmath import mpc
import time as time
import numpy as np
import matplotlib.pyplot as plt


import os
mp.mp.dps= 20
import RfuncCompSeries as RSeries
import RfuncCompCNCT as RCNTC

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 16})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
_mp = np.vectorize(mp.mpf)

#===============================================================================
# MAIN PLOT
#===============================================================================

def main_plot(saving = False):
    # Main plot:
    # Plots the mod-H function
    # the current with and without interference
    # and transmission
    genData = {
        "v":[mpf('8') * mpf(10**(3)), mpf('2.') * mpf(10**(3))],\
        "g":[1/mpf(8),1/mpf(8)],\
        "c":[1,1],\
        "a1":mpf('2.2')/ mpf(10**(6)), "a2":mpf('1.8')/mpf(10**(6)), \
        "T" :mpf(1) / mpf(10**(3)), "Q" :1/mpf(4)}  
    pfaff = {"Q":1/mpf(4),"g":[mpf(1)/mpf(8),mpf(1)/mpf(8)],"c":[1,1]}
    apf = {"Q":1/mpf(4), "g":[mpf(1)/mpf(8), mpf(3)/mpf(8)],"c":[-1,1]}
    
          
    mp.mp.dps = 60
    a = RSeries.RfuncMP(par = genData, nterms = 700,
                        Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 301))
    a.updateParameters(par = pfaff)
    a.genAnswer()
#    b = RSeries.RfuncMP(par = genData, nterms = 700,
#                        Vpoints = mp.linspace(0, mpf('1.')/mpf(10**4), 201))
#    b.updateParameters(par = apf)
#    b.genAnswer()

    
    fig = plt.figure()

    xt = np.linspace(0, 1 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]      
    
    ax2 = fig.add_subplot(211)  
    a.plotTotalCurrent(plotfig = ax2, label = r"With interference", 
                       linewidth=1.5)     
    a.plotSingleCurrent(plotfig = ax2, label = r'No interference', 
                        color = 'purple', linewidth=1.5)
                        
    ax2.get_lines()[1].set_dashes([5,2])    
    for i in ax2.get_lines(): i.set_color('black')
    
    ax2.set_title(r'Tunnelling current for the Pfaffian state')
    ax2.set_ylabel(r"$I_B/\mathrm{max}(I_B){}$")
    ax2.set_ybound([0,1])
    ax2.set_yticks([.25, .5, .75, 1])
    ax2.set_yticklabels([0.25, 0.5, 0.75, 1])
    ax2.set_xticks(xt)
    ax2.set_xticklabels([])

    ax2.legend(loc = 'upper right', prop={'size':12})
    plt.setp(ax2.get_xticklabels(), fontsize = 12.)
    plt.setp(ax2.get_yticklabels(), fontsize = 12.)    
    ax2.grid(True)
    
    
    ax3 = fig.add_subplot(212)  
    a.plotRfunction(plotfig = ax3, linewidth=1.5) 
    ax3.get_lines()[0].set_color('black')
    ax3.set_title(r'Modulating function for the Pfaffian state')
    ax3.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax3.set_xlabel(r'Volt [$\mu$V]')
    ax3.set_yticks([-0.25,0,.25,.5,.75,1])
    ax3.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax3.set_xticks(xt) 
    ax3.set_xticklabels(xt_labels)
    ax3.grid(True)
    plt.setp(ax3.get_xticklabels(), fontsize=12.)
    plt.setp(ax3.get_yticklabels(), fontsize=12.)        
    if saving: plt.savefig('main_plot.png', bbox_inches=0, dpi=fig.dpi)
    plt.show()
    return a
    
#===============================================================================
# TEMPERATURE    
#===============================================================================
    
def temperature(saving = False):
    generalInput = {
    "v":[mpf('3')* mpf(10**(4)),mpf('5.')*mpf(10**(3))],\
    "g":[1/mpf(8),1/mpf(8)],\
    "c":[1,1],\
    "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.5')/mpf(10**(6)), \
    "T" :mpf(10) / mpf(10**(3)), "Q" :1/mpf(4)}    
    
    

    #os.chdir('C:\\Data\\rfunc\\temp')
    names = [0,10,20]    
    temperatureset = [{"T":mpf(i)/mpf(10**3)} for i in names]
    a = ''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for j in range(3):
        i = temperatureset[j]
        if i["T"] < 10/mpf(10**3):
            mp.mp.dps = 60
            a = RSeries.RfuncMP(par = generalInput, nterms = 200)
        else:
            mp.mp.dps = 20
            a = RCNTC.RfuncCNTC(par = generalInput, nterms = 200000)
            mp.mp.dps = 20
        a.updateParameters(par = i)
            
        a.genAnswer()
        a.plotRfunction(plotfig = ax, label = str(names[j]) +' [mK]', linewidth=1.5) 
        mp.mp.dps = 20
        
        #del a.lda
        #del a
    
    dashstyle  = [(None, None), [7,2], [3,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i])    
    


    xt = np.linspace(0, 2 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]        
    ax.set_title(r'The modulating function \\ for different temperature scales')
    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax.set_xlabel(r'Volt [$\mu$V]')
    ax.set_yticks([-0.25,0,.25,.5,.75,1])
    ax.set_yticklabels([-0.25,0,0.25,0.5,0.75,1])
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.grid(True)


    ax.legend(loc='upper right', prop={'size':14})
    plt.setp(ax.get_xticklabels(), fontsize=12.)
    plt.setp(ax.get_yticklabels(), fontsize=12.)
    if saving: plt.savefig('temperature.png', bbox_inches=0, dpi=fig.dpi)
    plt.show()
    return a

#===============================================================================
# PLOTS FOR 5/2 STATE
#===============================================================================

def particles_5_2(saving = False):
    
    genData = {
        "v":[mpf('.9')* mpf(10**(4)),mpf('2.')*mpf(10**(3))],\
        "g":[1/mpf(8),1/mpf(8)],\
        "c":[1,1],\
        "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.5')/mpf(10**(6)), \
        "T" :mpf(1) / mpf(10**(3)), "Q" :1/mpf(4)}    
    
    pfaffE2 = {"Q":1/mpf(2), "g":[mpf(1)/mpf(2)],
               "c":[1], "v":[mpf('.9')*mpf(10**(4))]}
    pfaff = {"Q":1/mpf(4),"g":[mpf(1)/mpf(8), mpf(1)/mpf(8)], "c":[1,1]}
    apf = {"Q":1/mpf(4), "g":[mpf(1)/mpf(8), mpf(3)/mpf(8)], "c":[-1,1]}
    state331 = {"Q" :1/mpf(4), "g":[mpf(1)/mpf(8), mpf(1)/mpf(4)], "c":[1,1]}  
    particleset = [pfaff, apf, state331, pfaffE2]
    names = ["Pfaffian (e/4)", 
             "Anti-Pfaffian (e/4)",
            "(3,3,1)-state (e/4)", 
            "Laughlin (e/2)"]

    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    a = ''
    for i in range(len(names)):
        mp.mp.dps = 60
        a = RSeries.RfuncMP(par = genData, nterms = 800)
        #a = RCNTC.RfuncCNTC(par = genData, nterms = 100000)
        
        a.updateParameters(par = particleset[i])
            
        a.genAnswer()
        mp.mp.dps = 20
        a.plotRfunction(plotfig = ax, label = names[i]\
        , linewidth=1.5) 
        
        #del a.lda
        #del a
        
    dashstyle  = [(None, None), [10,4], [5,3,1,3], [2,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i])            
        
    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax.set_xlabel(r'Volt [$\mu$V]')
    ax.set_ybound([-.5, 1])
    ax.set_yticks([-0.25, 0, .25, .5, .75, 1])
    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    xt = np.linspace(0, 2 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.set_title(r"Modulating function for $\nu = 5/2 $ candidates")
    ax.legend(loc='upper right', prop={'size':14})
    ax.grid(True)    
    plt.setp(ax.get_xticklabels(), fontsize=12.)
    plt.setp(ax.get_yticklabels(), fontsize=12.)
    if saving: plt.savefig('particles_5_2.png', bbox_inches=0, dpi=fig.dpi)
    plt.show()
    return a
    

#===============================================================================
# PLOTS FOR 7/3 STATE
#===============================================================================


def particles_7_3(saving = False):
    
    genData = {
    "v":[mpf('.9')* mpf(10**(4)),mpf('2.')*mpf(10**(3))],\
    "g":[1/mpf(8),1/mpf(8)],\
    "c":[1,1],\
    "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.5')/mpf(10**(6)), \
    "T" :mpf(1) / mpf(10**(3)), "Q" :1/mpf(4)}    
        
    LaughlinE3 = {"Q":1/mpf(3),"g":[mpf(1)/mpf(2)],"c":[1],"v":[mpf('.9')*mpf(10**(4))]}
    aBS23 = {"Q":1/mpf(3),"g":[mpf(1)/mpf(3), mpf(5)/mpf(8)],"c":[1,1]}
    BS13 = {"Q":1/mpf(3), "g":[mpf(1)/mpf(3), mpf(3)/mpf(8)],"c":[-1,1]}
    aRR4 = {"Q" :1/mpf(6),"g":[mpf(1)/mpf(12), mpf(1)/mpf(4)], "c":[1,1]}  
    particleset = [aBS23, BS13, aRR4, LaughlinE3]
    names =[    r"$\overline{\mathrm{BS}}_{2/3}$ (e/3)",
                r"$\mathrm{BS}_{1/3}^{\psi}$ (e/3)",
                r"$\overline{\mathrm{RR}}_{k=4} $ (e/6)", 
                r"Laughlin (e/3)"]
    
    xt = np.linspace(0, 2 * 10**(-4), 5)
    xt_labels = [str(int(i * 10**6)) for i in xt]
    #os.chdir('C:\\Data\\rfunc\\temp')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    a = ''
    for i in range(0,len(particleset)):
        mp.mp.dps = 60
        a = RSeries.RfuncMP(par = genData, nterms = 700)
        #a = RCNTC.RfuncCNTC(par = genData, nterms = 100000)
        
        a.updateParameters(par = particleset[i])
            
        a.genAnswer()
        mp.mp.dps = 20
        a.plotRfunction(plotfig = ax, label = names[i]\
        , linewidth=1.5) 
        
        #del a.lda
        #del a

    dashstyle  = [(None, None), [10,4], [5,3,1,3], [2,4]]
    for i in range(len(ax.get_lines())):
        ax.get_lines()[i].set_color('black')
        ax.get_lines()[i].set_dashes(dashstyle[i]) 

    ax.set_ylabel(r"$\mathrm{Re}[H_{ij}^{\mathrm{mod}}]{}$")
    ax.set_xlabel(r'Volt [$\mu$V]')
    ax.set_ybound([-.5, 1])
    ax.set_yticks([-0.25, 0, .25, .5, .75, 1])
    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks(xt)
    ax.set_xticklabels(xt_labels)
    ax.set_title(r"Modulating function for $\nu = 7/3 $ candidates")
    ax.legend(loc='upper right', prop={'size':14})
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), fontsize=12.)
    plt.setp(ax.get_yticklabels(), fontsize=12.)
    if saving: plt.savefig('particles_7_3.png', bbox_inches=0, dpi=fig.dpi)
    plt.show()
    return a
    
def plot_all(saveon = False):
    os.chdir('C:\\Science\\LaTeX\\Interferometer PRB')
    main_plot(saving = saveon)
    temperature(saving = saveon)
    particles_5_2(saving = saveon)
    particles_7_3(saving = saveon)
#A = RCNTC.RfuncCNTC(par = genData, nterms = 100000)
#A.updateParameters(par =  {
#    "v":[mpf('5.')* mpf(10**(4)),mpf('5.')*mpf(10**(3))],\
#    "g":[1/mpf(8),1/mpf(8)],\
#    "c":[1,1],\
#    "a1":mpf('1.7')/ mpf(10**(6)),     "a2":mpf('1.3')/mpf(10**(6)), \
#    "T" :mpf(8) / mpf(10**(3)), "Q" :1/mpf(4)})
#A.updateParameters(par = pfaffData)
#A.genAnswer()
