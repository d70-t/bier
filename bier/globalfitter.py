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


from fitter import FieldModelFitter, LocalFitterMixin
import numpy as np

class GlobalFitter(FieldModelFitter, LocalFitterMixin):
    def _postInit(self):
        self.residuumArgs['fitGlobal'] = True
        self._localPostInit()
    def run(self, constantData=None, **kwargs):
        if constantData is not None:
            assert set(self.constantNames) == set(constantData.keys())
            self.constantData = {k:v[...,np.newaxis] for k,v in constantData.items()}
        else:
            self.constantData = {}
        popt, residual = self.doFit(self.field, self.guessVals, **self.constantData)
        return popt, residual
