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
    #_ = plotInterferenceAndFourier(saving = saveFigure) # 19 minutes
    #_ = plotDistanceEffectOnAB(saving = saveFigure) # 25 mins
    A = plotFrequencySpectrumWhenChangingDistance(saving = saveFigure) # 57 mins

    #_ = main_plot(saving = saveFigure) # 1 min
    #_ = temperature(saving = saveFigure) # 10.5 mins
    #_ = particles_5_2(saving = saveFigure) # 7 mins
    #_ = particles_7_3(saving = saveFigure) # 9 mins
    #_ = plotFrequency(saving = saveFigure) # 1.5 min     

    
    return A

if __name__ == '__main__':
    pass
#    matplotlib.rcParams['backend']
