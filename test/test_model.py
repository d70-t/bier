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


import bier.model as model

from unittest import TestCase
import sympy

class TestUtils(TestCase):
    def test_get_symbols_trivial(self):
        x = sympy.Symbol('x')
        self.assertEqual({x}, model.get_symbols(x))

    def test_get_symbols_two(self):
        x = sympy.Symbol('x')
        y = sympy.Symbol('y')
        z = sympy.Symbol('z')
        self.assertEqual({x, y}, model.get_symbols(x**2 + y + sympy.exp(x * y)))

