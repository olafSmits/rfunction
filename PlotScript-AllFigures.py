# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 10:54:28 2013
"""

import matplotlib
import matplotlib.pylab as plt
import cProfile
import timeit

# These functions plot the figures used in the article

from PlotScript_ABEffect_Frequency import plotInterferenceAndFourier
from PlotScript_ABEffect import plotDistanceEffectOnAB
from PlotScript_edgeVariation import plotFrequencySpectrumWhenChangingDistance
from PlotScript_frequencyAnalysis import plotFrequency
from PlotScript_voltVariation import main_plot, temperature
from PlotScript_voltVariation import particles_5_2, particles_7_3


plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size' : 16})
plt.rc('text', usetex=True)

def plotFigures(saveFigure = False):
    """ Plots all figures, with the option to save the figure.
    Also noted that time it took to compute each task, on a not-so-fast
    Vista laptop @ 2.2 GHz, 2 GB memory.
    """
    #_ = plotInterferenceAndFourier(saving = saveFigure) 
    #_ = plotDistanceEffectOnAB(saving = saveFigure) 
    A = plotFrequencySpectrumWhenChangingDistance(saving = saveFigure) 

    #_ = main_plot(saving = saveFigure) 
    #_ = temperature(saving = saveFigure) 
    #_ = particles_5_2(saving = saveFigure) 
    #_ = particles_7_3(saving = saveFigure) 
    #_ = plotFrequency(saving = saveFigure) 

    
    return A

if __name__ == '__main__':
    pass
#    matplotlib.rcParams['backend']
