# -*- coding: utf-8 -*-
"""
Created on Fri Feb 08 22:25:13 2013
"""
from __future__ import division
import numpy as np
import sympy.mpmath as mp
import matplotlib.pylab as plt
from numpy import newaxis, vectorize
from sympy.mpmath import mpf
from InputParameters import GLOBAL_TEMP, EMP
from Rfunc_series import Rfunc_series


fmfy = vectorize(mp.mpmathify)
fgamma = vectorize(mp.gamma)
freal = vectorize(mp.re)
fexp = vectorize(mp.exp)

pi = mpf(mp.pi)

class Rfunc_series_zeroT(Rfunc_series):
    T = 0
    def genGamma(self):
        trms = np.arange(0, self.nterms)
        self.gamma = fgamma(self.gtot) / fgamma(self.gtot + trms[:])
        Vp = np.power(-1j*self.Vq[newaxis,:], trms)
        self.gamma = self.gamma[:,newaxis] * Vp