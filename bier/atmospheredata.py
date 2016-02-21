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
import os

class AtmosphereData(object):
    def __init__(self, wvlns=None, folder=None):
        if folder is None:
            folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        self.folder = folder
        E_0 = np.loadtxt(os.path.join(self.folder, 'E0_sun.txt'),skiprows=11)
        if wvlns is None:
            self.wvlns = E_0[:,0]
            self.E_0 = E_0[:,1]
        else:
            self.wvlns = wvlns
            self.E_0 = np.interp(self.wvlns, E_0[:,0], E_0[:,1])
        a_o = np.loadtxt(os.path.join(self.folder, 'O2.A'),skiprows=4)
        self.a_o = np.interp(self.wvlns, a_o[:,0], a_o[:,1])
        a_oz = np.loadtxt(os.path.join(self.folder, 'O3.A'),skiprows=4)
        self.a_oz = np.interp(self.wvlns, a_oz[:,0], a_oz[:,1])
        a_wv = np.loadtxt(os.path.join(self.folder, 'WV.A'),skiprows=4)
        self.a_wv = np.interp(self.wvlns, a_wv[:,0], a_wv[:,1])
