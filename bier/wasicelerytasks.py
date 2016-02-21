# -*- coding: utf-8 -*-
#    This program is part of BIER (Basic Irradiance Estimation and Regression).
#    Copyright (C) 2016 Tobias Kölling <tobias.koelling@physik.uni-muenchen.de>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import absolute_import

from celeryapp import app
from modelcache import ModelCache
import scipy.optimize

mc = ModelCache()

@app.task
def doFit(point, spectrum, guess, wvlns, fixedValues, method, weights=None, limVals=None, constantData=None):
    if constantData is None:
        constantData = {}
    if method == 'curve_fit':
        syms, func = mc.getModel(wvlns, fixedValues, 'E', weights, constantFieldNames=constantData.keys())
        if len(constantData) > 0:
            func = func(**constantData)
        popt, pcov = scipy.optimize.curve_fit(func, wvlns, spectrum, guess,ftol=1e-9,xtol=1e-9,maxfev=10000)
    elif method == 'tnc':
        syms, func = mc.getModel(wvlns, fixedValues, 'R,dR', weights, constantFieldNames=constantData.keys())
        if len(constantData) > 0:
            func = func(**constantData)
        popt,nfeval,rc = scipy.optimize.fmin_tnc(func, guess, args=(spectrum,), bounds=limVals, messages=0)
    elif method == 'downhill_simplex':
        syms, func = mc.getModel(wvlns, fixedValues, 'R', weights, constantFieldNames=constantData.keys())
        if len(constantData) > 0:
            func = func(**constantData)
        popt = scipy.optimize.fmin(func, np.array(guess), args=(spectrum,), disp=0)
    return 'ok', (point, popt)
