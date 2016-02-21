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
import re
import datetime
import dateutil
from runmacs.spec.util.decorators import lazyprop
from runmacs.processor.utils import rotAzEl, rotAzElRev, constrainAzEl
from runmacs.spec.io.envi import read_envi_header, parseIsoTime

dateRe = re.compile('date:\\s*([0-9]+(?:\\.[0-9]+)?)')
class WasiEnvi(object):
    def __init__(self, fn, ureg=None):
        if fn[-4:] in ('.hdr', '.raw'):
            fn = fn[:-4]
        header = read_envi_header(fn + '.hdr')
        samples = int(header['samples'])
        lines = int(header['lines'])
        bands = int(header['bands'])
        rawShape = (lines, bands, samples)
        self.shape = (rawShape[0], rawShape[2], rawShape[1])
        az = np.deg2rad(float(header['sun azimuth']))
        el = np.deg2rad(float(header['sun elevation']))
        self.sunPosition = (az, el)
        try:
            self.date = parseIsoTime(header['acquisition time'])
        except AttributeError:
            self.date = None
        self.refaz, self.refel, self.azmin, self.elmin, self.azstep, self.elstep = map(float,header['map info'][1:7])
        self.azmin, self.elmin, self.azstep, self.elstep = map(np.deg2rad, (self.azmin, self.elmin, self.azstep, self.elstep))
        self.azmin0 = self.azmin-((self.refaz-1)*self.azstep)
        self.elmin0 = self.elmin-((self.refel-1)*self.elstep)
        self.grid = coordinates = np.mgrid[
                            self.elmin0:self.elmin0+self.elstep*(self.shape[0]-1):self.shape[0]*1j,
                            self.azmin0:self.azmin0+self.azstep*(self.shape[1]-1):self.shape[1]*1j][::-1]
        try:
            self.wvlns = map(float, header['wavelength'])
        except KeyError:
            self.wvlns = None
        self.data = np.memmap(fn+'.raw',shape=rawShape,dtype='float32').transpose(0,2,1)
        reSliceData = [slice(None), slice(None)]
        if self.grid[0,0,-1] < self.grid[0,0,0]:
            reSliceData[1] = slice(None, None, -1)
        if self.grid[1,-1,0] < self.grid[1,0,0]:
            reSliceData[0] = slice(None, None, -1)
        self.grid = self.grid[tuple([slice(None)] + reSliceData)]
        self.data = self.data[tuple(reSliceData)]
        try:
            self.units = header['data units']
            if ureg is not None:
                self.units = ureg(self.units)
        except KeyError:
            self.units = None
    def __repr__(self):
        return "<WasiEnvi, %s, %s, %s>"%(self.date, self.shape, self.sunPosition)
    @property
    def az(self):
        return self.grid[0]
    @property
    def el(self):
        return self.grid[1]
    @lazyprop
    def shv(self):
        rotDelta = (self.sunPosition[0]+np.pi, 1.5*np.pi-self.sunPosition[1])
        doRot = lambda c: constrainAzEl(*rotAzElRev(c, rotDelta, mirror=True))
        return np.array(map(doRot,self.grid.transpose(1,2,0).reshape((self.shape[0]*self.shape[1], 2)))).reshape((self.shape[0], self.shape[1], 2)).transpose(2,0,1)
    @lazyprop
    def sca(self):
        return np.pi/2-self.shv[1]
    @lazyprop
    def rshv(self):
        rotDelta = (self.sunPosition[0], 1.5*np.pi-self.sunPosition[1])
        doRot = lambda c: constrainAzEl(*rotAzElRev(c, rotDelta, mirror=True))
        return np.array(map(doRot,self.grid.transpose(1,2,0).reshape((self.shape[0]*self.shape[1], 2)))).reshape((self.shape[0], self.shape[1], 2)).transpose(2,0,1)
    @lazyprop
    def rsca(self):
        return np.pi/2-self.rshv[1]
