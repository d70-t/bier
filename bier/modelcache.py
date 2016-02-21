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


from atmospheredata import AtmosphereData
from downwelling import DownwellingModel
import numpy as np

class ModelCache(object):
    def __init__(self):
        self.cache = {}
    def getModel(self, wvlns, fixedValues, requestedFunction, weights=None, constantFieldNames=None):
        if requestedFunction == 'E':
            weights = None
        if weights is not None:
            weights = tuple(weights)
        if constantFieldNames is not None:
            constantFieldNames = tuple(sorted(constantFieldNames))
        key = (tuple(wvlns), weights, tuple((k,v) for k,v in sorted(fixedValues.items())), constantFieldNames, requestedFunction)
        try:
            return self.cache[key]
        except KeyError:
            m = DownwellingModel(AtmosphereData(wvlns=wvlns))
            for k,v in fixedValues.items():
                m.setConstant(k,v)
            if constantFieldNames is not None:
                for n in constantFieldNames:
                    m.setConstantField(n)
            if requestedFunction == 'E':
                f = m.getCompiledE()
            elif requestedFunction == 'R':
                f = m.getCompiledResiduumFunction(includeDerrivative=False,weights=np.array(weights))
            elif requestedFunction == 'R,dR':
                f = m.getCompiledResiduumFunction(includeDerrivative=True,weights=np.array(weights))
            else:
                raise ValueError('model function ist not available')
            self.cache[key] = f
            return f
