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


import numpy as np

def fillModelConstantsFromSourceAndTaskDef(model, source, taskDef):
    takenConstants = set()
    if taskDef['constants'].get('theta_sun', False) is True:
        model.setConstant('theta_sun', np.pi/2 - source.sunPosition[1])
        takenConstants.add('theta_sun')
    constantFields = {}
    for n in ['az', 'el', 'sca', 'rsca']:
        if taskDef['constants'].get(n, False) is True:
            constantFields[n] = getattr(source, n)
            takenConstants.add(n)
    map(model.setConstantField, constantFields.keys())
    for k, v in taskDef['constants'].items():
        if k not in takenConstants:
            model.setConstant(k, float(v))
    return constantFields
