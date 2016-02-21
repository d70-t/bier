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
"""
Functions helping with geometric calculations
"""

import numpy as np

def constrain_az_el(az_, el_):
    """
    Constrains azimuth and elevation angles.

    The resulting ranges will be:
    * azimuth: 0 ... 2*π
    * elevation: -π/2 ... π/2
    """
    el = ((el_ + np.pi/2.)%(2.*np.pi)) - (np.pi/2.)
    el_mask = el > (np.pi/2.)
    el = np.where(el_mask, np.pi-el, el)
    az = np.where(el_mask, az_+np.pi, az_)%(2.*np.pi)
    return az, el

def Rx(alpha):  # pylint: disable=invalid-name
    """
    create rotation matrix around x axis

    :note: alpha can be an array of any shape
    """
    alpha = np.array(alpha)
    res = np.zeros(alpha.shape + (3, 3))
    res[..., 0, 0] = 1
    res[..., 1, 1] = np.cos(alpha)
    res[..., 1, 2] = -np.sin(alpha)
    res[..., 2, 1] = np.sin(alpha)
    res[..., 2, 2] = np.cos(alpha)
    return res
def Ry(alpha):  # pylint: disable=invalid-name
    """
    create rotation matrix around y axis

    :note: alpha can be an array of any shape
    """
    alpha = np.array(alpha)
    res = np.zeros(alpha.shape + (3, 3))
    res[..., 1, 1] = 1
    res[..., 0, 0] = np.cos(alpha)
    res[..., 0, 2] = np.sin(alpha)
    res[..., 2, 0] = -np.sin(alpha)
    res[..., 2, 2] = np.cos(alpha)
    return res
def Rz(alpha):  # pylint: disable=invalid-name
    """
    create rotation matrix around z axis

    :note: alpha can be an array of any shape
    """
    alpha = np.array(alpha)
    res = np.zeros(alpha.shape + (3, 3))
    res[..., 2, 2] = 1
    res[..., 0, 0] = np.cos(alpha)
    res[..., 0, 1] = -np.sin(alpha)
    res[..., 1, 0] = np.sin(alpha)
    res[..., 1, 1] = np.cos(alpha)
    return res

def shapeBroadcastCompare(a, b):
    if len(a) != len(b):
        return False
    for ai, bi in zip(a, b):
        if ai != bi and ai != 1 and bi != 1:
            return False
    return True

def diagDot(r1, r2):
    """
    Dot product for last two axes, all prepended axes must have
    the same shape and the diagonal is returned.
    If the shape indicates that one of the arguments contains a vector and not a matrix,
    rules of vector-matrix multiplication are applied.

    This is what is needed for the dot product of "lists" of rotation matrices.
    """
    if len(r1.shape) >= 2 \
            and len(r2.shape) >= 2 \
            and shapeBroadcastCompare(r1.shape[:-2], r2.shape[:-2]) \
            and r1.shape[-1] == r2.shape[-2]:
        return np.einsum('...ij,...jk->...ik', r1, r2)
    elif len(r1.shape) >= 2 \
            and len(r2.shape) >= 1 \
            and shapeBroadcastCompare(r1.shape[:-2], r2.shape[:-1]) \
            and r1.shape[-1] == r2.shape[-1]:
        return np.einsum('...ij,...j->...i', r1, r2)
    elif len(r1.shape) >= 1 \
            and len(r2.shape) >= 2 \
            and shapeBroadcastCompare(r1.shape[:-1], r2.shape[:-2]) \
            and r1.shape[-1] == r2.shape[-2]:
        return np.einsum('...j,...jk->...k', r1, r2)
    raise ValueError('shape missmatch %s and %s'%(r1.shape, r2.shape))

def azel2ned(azel):
    az, el = azel
    return np.array((np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), -np.sin(el)))
def ned2azel(ned):
    n, e, d = ned
    s = d/((ned**2).sum(axis=0))
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    el = -np.arcsin(np.where(s < -1., -1., np.where(s > 1., 1., s)))
    return np.array((np.arctan2(e, n)%(np.pi*2), el))


def rot_az_el(inp, delta, mirror=False):
    """
    Makes a forward rotation in azel coordinates (e.g. sun-system to observer system)
    :param inp:nd-array with az,el as 0-th axis
    """
    R = diagDot(Rz(delta[0]), Ry(delta[1]))
    if mirror:
        R = -R
    return ned2azel(diagDot(R, azel2ned(inp)))
def rot_az_el_rev(inp, delta, mirror=False):
    """
    Makes a reverse rotation in azel coordinates (e.g. observer system to sun system)
    :param inp:nd-array with az,el as 0-th axis
    """
    R = diagDot(Ry(-delta[1]), Rz(-delta[0]))
    if mirror:
        R = -R
    return ned2azel(diagDot(R, azel2ned(inp)))


