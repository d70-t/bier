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


import sympy
from bier.model import AnalyticModel


# pylint will produce tons of errors for these mathematical names, in this
# case, I would argue that they are ok
# pylint: disable=invalid-name

#tabulated;
E_0 = sympy.Symbol('E_0')
a_o = sympy.Symbol('a_o')
a_oz = sympy.Symbol('a_oz')
a_wv = sympy.Symbol('a_wv')


theta_sun = sympy.Symbol('theta_sun')
el = sympy.Symbol('el') #viewing elevation
sca = sympy.Symbol('sca') #scattering angle from the sun
rsca = sympy.Symbol('rsca') #reverse scattering angle from the sun
t = sympy.Symbol('t') #rayleigh weighting parameter
s = sympy.Symbol('s') #rayleigh sun weighting parameter

gfs = sympy.Symbol('gfs') #aerosol forwardscatter anisotropy
bs = sympy.Symbol('bs') #aerosol backscatter
gbs = sympy.Symbol('gbs') #aerosol backscatter anisotropy

wvl = sympy.Symbol('wvl')
P = sympy.Symbol('P')
alpha = sympy.Symbol('alpha')
wvl_a = sympy.Symbol('wvl_a') #typically 550
beta = sympy.Symbol('beta')
AM = sympy.Symbol('AM') #air mass type 1...10
RH = sympy.Symbol('RH') #relative humidity

H_oz = sympy.Symbol('H_oz')
WV = sympy.Symbol('WV')

hsc = sympy.Symbol('hsc') #mean scattering height
zsc = sympy.Symbol('zsc') #barometric scale height

f_dd = sympy.Symbol('f_dd')
f_ds = sympy.Symbol('f_ds')

g_dsr = sympy.Symbol('g_dsr')
g_dsa = sympy.Symbol('g_dsa')

def pathLengthGC(theta):
    """
    Atmospheric path length following Gregg and Cader
    """
    a = 0.15
    b = 3.885*sympy.pi/180.
    c = 1.253
    return 1/(sympy.cos(theta) + a*(90+b-theta*(180./sympy.pi))**(-c))

def pathLengthKY(theta):
    """
    Atmospheric path length following Kasten and Young
    """
    a = 0.50572
    b = 6.07995
    c = 1.6364
    return 1/(sympy.cos(theta) + a*(90+b-theta*(180./sympy.pi))**(-c))

def pathLengthOzone(theta):
    """
    Atmospheric path length for ozone calculations
    """
    return 1.0035/(sympy.cos(theta)**2 + .007)**.5

pathLength = pathLengthKY

def ssPathLength(thetaView, thetaSun, meanScatterHeight, plImpl=pathLength):
    """
    Computes total path length of a photon with a single scattering
    at a given mean scattering height.

    :param thetaView: zenith angle of viewing direction
    :param thetaSun: zenith angle of light source (e.g. sun)
    :param meanScatterHeight: density weighted mean height of scattering events [0(=Earth Surface)...1(=TOA)]
    :param plImpl: implementation of atmospheric pathLength
    """
    return meanScatterHeight * plImpl(thetaView) + (1-meanScatterHeight) * plImpl(thetaSun)

def pathLengthPressureCorrection(M, P):
    """
    Computes pressure correction for path length

    :param M: uncorrected path length
    :param P: pressure in millibar
    """
    return M * (P / 1013.25)

theta_view = sympy.pi/2. - el
M = ssPathLength(theta_view, theta_sun, hsc)
M_bar = pathLengthPressureCorrection(M, P)

M_sun = pathLength(theta_sun)
M_direct = pathLength(theta_view)
M_direct_bar = pathLengthPressureCorrection(M_direct, P)

#M_oz = pathLengthOzone(theta_sun)
M_oz = pathLengthOzone(theta_view)
M_a = M_direct
M_o = M_direct_bar
M_wv = M_direct

#g = sympy.Max(sympy.Min(-.1417*alpha+.82,.82),.65)
##g = sympy.Piecewise((.82, alpha<0.0), (-.1417*alpha+.82, alpha<=1.2), (0.65, True))
##g = -.1417*alpha+.82
#B_3 = sympy.ln(1-g)
#B_1 = B_3*(1.459 + B_3*(.1595 + .4129*B_3))
#B_2 = B_3*(.0783 + B_3*(-.3824 - .5874*B_3))
#F_a = 1 - .5*sympy.exp((B_1+B_2*sympy.cos(theta_sun))*sympy.cos(theta_sun))

def phaseRayleigh(sca):
    """
    Rayleigh phase function
    """
    return .75 * (1. + sympy.cos(sca)**2)

def phaseHG(sca, g):
    """
    henyey greenstein phase funcion
    """
    return (1./4.*sympy.pi) * (1. - g**2) / ((1 + g**2) - (2*g) * sympy.cos(sca))**1.5

#aerosol
tau_a = beta*(wvl/wvl_a)**(-alpha)
omega_a = (-.0032*AM + .972) * sympy.exp(3.06e-4*100*RH)

#transmission
tau_r = 1. / (115.6406*(wvl/1000.)**4 - 1.335*(wvl/1000.)**2)
T_r = sympy.exp(-M_bar * tau_r)
#T_r = sympy.exp(-M_bar * tau_r)
#gen_rayleigh = f_rayleigh * M_rayleigh * (1-sympy.exp(-M_bar * tau_r)**.95)
#gen_rayleigh = f_rayleigh * M_rayleigh * (1-T_r**.95)
T_aa = sympy.exp(-(1-omega_a)*tau_a*M_a)
T_as = sympy.exp(-omega_a*tau_a*M_a)
#gen_aerosol = f_aerosol * M_aerosol * (1-T_as)
tau_oz = a_oz*H_oz
T_oz = sympy.exp(-tau_oz*M_oz)
T_o = sympy.exp((-1.41*a_o*M_o)/(1 + 118.3*a_o*M_o)**.45)
T_wv = sympy.exp((-0.2385*a_wv*WV*M_wv)/(1 + 20.07*a_wv*WV*M_wv)**.45)

tau_transmission = tau_r + tau_a + tau_oz

#scattering

Pr = phaseRayleigh(sca) * M_direct_bar * tau_r
Pa = ((phaseHG(sca, gfs) + bs * phaseHG(rsca, gbs)) / (1 + bs)) * M_direct * tau_a

#radiances # earth-sun distance is not covered
E_dd = T_r * T_aa * T_as * T_oz * T_o * T_wv
E_dsr = T_r * T_aa * T_as * T_oz * T_o * T_wv * Pr
E_dsa = T_r * T_aa * T_as * T_oz * T_o * T_wv * Pa

def T_barom_scale(tau, M_v, M_s):
    mu_s = 1. / M_s
    mu_v = 1. / M_v
    normal = zsc * mu_v * mu_s * (sympy.exp(-(tau/mu_v + 1./zsc) - sympy.exp(-(tau/mu_s)))) / (tau * zsc * (mu_v - mu_s) - mu_v * mu_s)
    special = (sympy.exp(1./zsc) - 1.) * zsc * sympy.exp(-(tau/mu_v + 1./zsc))
    return sympy.Piecewise((normal, mu_v != mu_s), (special, True))

def T_barom(tau, M_v, M_s):
    mu_s = 1. / M_s
    mu_v = 1. / M_v
    normal = mu_v * mu_s * (sympy.exp(-((tau + mu_v)/mu_v)) - sympy.exp(-(tau/mu_s))) / (tau * (mu_v - mu_s) - mu_v * mu_s)
    special = (sympy.exp(1) - 1) * sympy.exp(-(tau + mu_v) / (mu_v))
    return sympy.Piecewise((normal, mu_v != mu_s), (special, True))

def T_constant(tau, M_v, M_s):
    mu_s = 1. / M_s
    mu_v = 1. / M_v
    normal = (mu_s / tau) * (sympy.exp(-tau / mu_s) - sympy.exp(-tau / mu_v)) / (4 * sympy.pi * (mu_s - mu_v))
    special = sympy.exp(-tau / mu_v) / (4 * sympy.pi * mu_v)
    return sympy.Piecewise((normal, mu_v != mu_s), (special, True))

T_impl = T_constant

def L_New(E_0, omega_0, P, tau, tau_sc, M_v, M_s):
    return E_0 * omega_0 * P * tau_sc * T_impl(tau, M_v, M_s)

def L_Full(E_0, omega_0, P, tau, tau_sc, M_v, M_s):
    mu_0 = 1. / M_s
    mu = 1. / M_v
    normal =  E_0 * mu_0 * omega_0 * P * (tau_sc / tau) * (sympy.exp(-tau / mu_0) - sympy.exp(-tau / mu)) / (4 * sympy.pi * (mu_0 - mu))
    special = E_0 * omega_0 * P * sympy.exp(-tau / mu_0) * tau_sc / (4 * sympy.pi * mu)
    return sympy.Piecewise((normal, mu_0 != mu),(special, True))

def L_MFull(E_0, omega_0, P, tau, tau_sc, M, M_0):
    mu_0 = 1. / M_0
    mu = 1. / M
    normal =  E_0 * mu_0 * omega_0 * P * (sympy.exp(-tau / mu_0) - sympy.exp(-tau / mu)) / (4 * sympy.pi * (mu_0 - mu))
    special = E_0 * omega_0 * P * sympy.exp(-tau / mu_0) * tau / (4 * sympy.pi * mu)
    return sympy.Piecewise((normal, mu_0 != mu),(special, True))

def L_MSimplified(E_0, omega_0, P, tau, tau_sc, M, M_0):
    return E_0 * omega_0 * P * tau_sc * M / (4 * sympy.pi)

T_residual = T_o * T_wv

L_Model = L_New

L_dsr = L_Model(E_0, 1, phaseRayleigh(sca), tau_transmission, tau_r, M_direct, M_sun)
L_dsa = L_Model(E_0, omega_a, phaseHG(sca, gfs), tau_transmission, tau_a, M_direct, M_sun)

L_ds = T_residual * (g_dsr * L_dsr + g_dsa * L_dsa)
L_d = f_dd * E_dd + f_ds * L_ds

def DownwellingModel():
    return AnalyticModel(L_d)

