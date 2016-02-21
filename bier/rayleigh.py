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

def pathLength(elevation):
    theta = np.pi/2 - elevation
    a=.50572;b=6.07995;c=1.6364
    return 1/(np.cos(theta) + a*(90+b-theta*(180./np.pi))**(-c))

def verticalRayleighCoefficient(el, t=.2):
    pl = pathLength(el)
    return pl*np.exp(-t*pl)

def sca2rayleigh(sca, el, e=0.013742606681169235, t=.2):
    fR = 1 + np.cos(sca)**2
    # basically some factor * rayleigh creation * rayleigh attenuation
    return e*fR*verticalRayleighCoefficient(el, t)
