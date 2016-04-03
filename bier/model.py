# -*- coding: utf-8  -*-
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
import numpy as np
from bier.atmospheredata import AtmosphereData
import functools

import theano
import theano.tensor as T

def get_symbols(expression):
    """
    Extracts a ``set`` of all used symbols from an arbitrary sympy ``expression``.
    """
    symbols = set()
    for symbol_candidate in sympy.utilities.postorder_traversal(expression):
        if isinstance(symbol_candidate, sympy.Symbol):
            symbols.add(symbol_candidate)
    return symbols


def piecewise2theano(*args):
    """
    Converts a sympy piecewise defined function to a theano expression.
    """
    #each arg is (value, condition)
    if len(args) < 1:
        raise ValueError('must get at least one arg')
    if len(args) == 1:
        return args[0][0]
    else:
        return T.switch(args[0][1], args[0][0], piecewise2theano(*args[1:])) #pylint: disable=star-args


class AnalyticModel(object):
    """
    An analytic model model is created from a sympy ``expression``.

    :param expression: sympy expression desctibing the model.
    :param atmosphere: atmospheric parameters (1nm Earth Atmosphere if None)
    :param presision: Compute precision, either 'float32' or 'float64'
    """
    def __init__(self, expression, atmosphere=None, precision='float64'):
        if atmosphere is None:
            self.atmosphere = AtmosphereData()
        else:
            self.atmosphere = atmosphere
        self._E = expression
        self.precision = precision
        self.fixedValues = {}
        self.constantFields = set()

    def setConstant(self, name, value):
        """
        Sets the symbol ``name`` to a given ``value``.

        ``value`` cn either be an numpy array or a scalar value.
        """
        if name in self.fixedValues:
            return
        self.fixedValues[name] = value
        if isinstance(value, np.ndarray):
            value = value.astype(self.precision)
        else:
            value = np.array([value], dtype=self.precision)
        self._E = self._E.subs(sympy.Symbol(name), value)

    def setConstantField(self, name):
        self.constantFields.add(name)

    @property
    def E(self):
        return self._E

    @property
    def E_numeric(self):
        return self.E.subs(sympy.pi, np.array(np.pi, dtype=self.precision))

    def getSymbols(self):
        """
        Gets symbols of the current model.

        Returns a tuple consisting of::

            * symbolic representation of the model
            * symbols of atmospheric parameters
            * symbols set to constants
            * free symbols which will be subject to a fit
        """
        symbolic = self.E_numeric
        symbols = get_symbols(symbolic)
        atmosphereParams = map(sympy.Symbol, ['E_0', 'a_o', 'a_oz', 'a_wv', 'wvl'])
        constantSymbols = map(sympy.Symbol, sorted(self.constantFields))
        freeSymbols = sorted([s for s in symbols if s not in atmosphereParams + constantSymbols], key=lambda x: x.name)
        return symbolic, atmosphereParams, constantSymbols, freeSymbols

    def getFreeParamNames(self):
        symbolic, atmosphereParams, constantSymbols, freeSymbols = self.getSymbols()
        return [s.name for s in freeSymbols]

    def getConstantParamNames(self):
        symbolic, atmosphereParams, constantSymbols, freeSymbols = self.getSymbols()
        return [s.name for s in constantSymbols]

    def getLambdifiedE(self):
        symbolic, atmosphereParams, constantSymbols, freeSymbols = self.getSymbols()
        symbols = atmosphereParams + constantSymbols + freeSymbols
        #la1 = ufuncify(symbols, symbolic)
        la1 = sympy.lambdify(symbols, symbolic, "numpy")
        return ([s.name for s in freeSymbols],
                functools.partial(la1, self.atmosphere.E_0, self.atmosphere.a_o, self.atmosphere.a_oz, self.atmosphere.a_wv))

    def getTheanoE(self, useFields=False, fitGlobal=False):
        theano.config.floatX = self.precision
        symbolic, atmosphereParams, constantSymbols, freeSymbols = self.getSymbols()
        symbols = atmosphereParams + constantSymbols + freeSymbols

        self.loadedFields = [theano.shared(x.astype(self.precision),name=s.name) for x,s in zip([self.atmosphere.E_0,
                                                                    self.atmosphere.a_o,
                                                                    self.atmosphere.a_oz,
                                                                    self.atmosphere.a_wv,
                                                                    self.atmosphere.wvlns],atmosphereParams)]
        FieldType = T.TensorType(self.precision, (False, False, True))
        ScalarType = T.TensorType(self.precision, ())
        if useFields:
            convertConstantSymbol = lambda x: FieldType(x.name)
            convertFreeSymbol = lambda x: FieldType(x.name)
        else:
            if fitGlobal:
                convertConstantSymbol = lambda x: FieldType(x.name)
            else:
                convertConstantSymbol = lambda x: ScalarType(x.name)
            convertFreeSymbol = lambda x: ScalarType(x.name)
        convertedSymbols = map(convertConstantSymbol, constantSymbols) + map(convertFreeSymbol, freeSymbols)
        la1 = sympy.lambdify(symbols, symbolic, [T, {'Min': T.minimum,
                                                     'Max': T.maximum,
                                                     'Piecewise': piecewise2theano}])
        theano_E = la1(*(self.loadedFields+convertedSymbols))
        return theano_E, convertedSymbols

    def getCompiledE(self, useFields=False, fitGlobal=False):
        constantsNames = self.getConstantParamNames()
        E, syms = self.getTheanoE(useFields, fitGlobal)
        _func = theano.function(syms, E, on_unused_input='ignore')
        if len(constantsNames) == 0 or useFields or fitGlobal:
            func = _func
        else:
            def func(**constants):
                def __func(x):
                    return _func(*([constants[n] for n in constantsNames] + list(x)))
                return __func
        return [x.name for x in syms], func

    def getCompiledResiduumFunction(self,includeDerrivative=True,method='lesq',weights=None, fitGlobal=False):
        """
        Generates a residuum function with derrivatives, compatible to scipy.optimize.fmin_tnc

        :note: when fitGlobal is given, weights must provide weights including the volume element of integration
        """
        E, syms = self.getTheanoE(useFields=False, fitGlobal=fitGlobal)
        if fitGlobal:
            reference = T.TensorType(self.precision, (False, False, False))()
        else:
            reference = T.TensorType(self.precision, (False, ))()
        if weights is not None:
            normF = 1./np.sum(weights)
            weights = theano.shared(weights.astype(self.precision),name='weights')
        else:
            weights = 1
            normF = 1./len(self.atmosphere.wvlns)
        if method == 'lesq':
            R = T.sum(weights * ((E-reference)**2)) * normF
        else:
            raise ValueError('method "%s" is not defined'%method)
        constantsNames = self.getConstantParamNames()
        freeSymbols = syms[len(constantsNames):]
        print ['reference'] + syms
        if includeDerrivative:
            dR = T.grad(R, freeSymbols)
            _residuum = theano.function([reference]+syms, [R]+dR, on_unused_input='ignore')
            if len(constantsNames) == 0:
                def residuum(x, ref):
                    res = _residuum(ref, *x)
                    return res[0], res[1:]
            else:
                def residuum(**constants):
                    constants = [constants[n] for n in constantsNames]
                    def residuumImpl(x, ref):
                        res = _residuum(ref, *(constants + list(x)))
                        return res[0], res[1:]
                    return residuumImpl
            return [x.name for x in syms], residuum
        else:
            _residuum = theano.function([reference]+syms, R, on_unused_input='ignore')
            if len(constantsNames) == 0:
                def residuum(x, ref):
                    return _residuum(ref, *x)
            else:
                def residuum(**constants):
                    constants = [constants[n] for n in constantsNames]
                    def residuumImpl(x, ref):
                        return _residuum(ref, *(constants + list(x)))
                    return residuumImpl
            return [x.name for x in syms], residuum

