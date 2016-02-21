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
import itertools

class DidNotConverge(Exception):
    def __init__(self, reason):
        self.reason = reason
        super(DidNotConverge, self).__init__(reason)

class FieldModelFitter(object):
    def __init__(self, field, wvlns, model, guess, lims, weights=None, method='tnc', residuumArgs={}):
        #wvlns must be on last axis of field
        self.field = field
        self.wvlns = wvlns
        self.model = model
        self.constantNames = self.model.getConstantParamNames()
        self.method = method
        self.residuumArgs = residuumArgs.copy()
        self.residuumArgs['weights'] = weights
        self.guess = guess
        self.lims = lims
        self.weights = weights
        self._postInit()
        self.paramNames = self.model.getFreeParamNames()
        self.guessVals = [guess[n] for n in self.paramNames]
        self.limVals = np.array([lims[n] for n in self.paramNames])
    def _postInit(self):
        pass

class PointwiseFieldModelFitter(FieldModelFitter):
    def run(self, useNeighbors=True, progressBar=None, constantData=None):
        if constantData is not None:
            assert set(self.constantNames) == set(constantData.keys())
            self.constantData = constantData
        else:
            self.constantData = {}
        self.useNeighbors = useNeighbors
        self.progressBar = progressBar
        resShape = self.field.shape[:-1]
        self.openPoints = set(list(itertools.product(*map(xrange, resShape))))
        self.totalCount = float(len(self.openPoints))
        self.nextPoints = {self.openPoints.pop(): self.guessVals for _ in xrange(50)} #seed N starting points
        self.processingPoints = {}
        self.solvedPoints = {}
        
        self._prepareRun()

        try:
            while len(self.openPoints) > 0 and (len(self.processingPoints) > 0 or len(self.nextPoints) > 0):
                self._sendMorePoints()
                self._receivePoints()
            while len(self.processingPoints) > 0:
                self._receivePoints()
        finally:
            self._cleanupRun()
        print 'done'
        result = np.zeros(resShape + (len(self.paramNames),))
        residual = np.zeros(resShape)
        for point, resValues in self.solvedPoints.items():
            result[point], residual[point] = resValues
        return result, residual
    def _prepareRun(self):
        pass
    def _cleanupRun(self):
        pass
    def _sendMorePoints(self):
        for point, guess in self.nextPoints.items():
            #print 'sending', point
            self._queuePoint(point, guess, {k:v[point] for k,v in self.constantData.items()})
            self.processingPoints[point] = guess
        self.nextPoints.clear()
        if self.progressBar is None:
            print len(self.openPoints), 'to go'
        else:
            self.progressBar.updateProgress(1.-(len(self.openPoints)/self.totalCount))
    def _markSolved(self, point, fitValues, residual):
        if point in self.solvedPoints:
            return
        self.solvedPoints[point] = fitValues, residual
        del self.processingPoints[point]
        neighbors = list(itertools.product(*[[e+o for o in (-1,+1)] for e in point]))
        if not self.useNeighbors:
            fitValues = self.guessVals
        else:
            nfV = np.array(fitValues)
            if not all((nfV>=self.limVals[:,0]) & (nfV <= self.limVals[:,1])):
                print 'res out of bounds:', point, nfV
                fitValues = self.guessVals
        for n in neighbors:
            try:
                self.openPoints.remove(n)
            except KeyError:
                pass
            else:
                if n not in self.nextPoints:
                    self.nextPoints[n] = fitValues
    def _markFailed(self, point, reason):
        try:
            del self.processingPoints[point]
            print "error", point, reason
        except KeyError:
            pass

import scipy.optimize
methodMap = {'downhill_simplex': {'needJac': False, 'scipyName': 'Nelder-Mead'},
             'tnc': {'needJac': True, 'scipyName': 'TNC'},
             }

class LocalFitterMixin(object):
    def _localPostInit(self):
        if self.method == 'curve_fit':
            _, self.fitfunc = self.model.getLambdifiedE()
            def doFit(spectrum, guess, **constantData):
                if len(constantData) > 0:
                    ff = self.fitfunc(**constantData)
                else:
                    ff = self.fitfunc
                popt, pcov = scipy.optimize.curve_fit(ff, self.wvlns, spectrum, guess,ftol=1e-9,xtol=1e-9,maxfev=10000)
                return popt, np.nan #residual is not available here
        else:
            if self.method.startswith('basinhopping:'):
                localMethod = self.method.split(':')[1]
                basinhopping = True
            else:
                localMethod = self.method
                basinhopping = False
            try:
                methodInfo = methodMap[localMethod]
            except KeyError:
                raise ValueError('fit method "%s" not known!'%localMethod)
            method = methodInfo['scipyName']
            needJac = methodInfo['needJac']
            _, self.fitfunc = self.model.getCompiledResiduumFunction(includeDerrivative=needJac,**self.residuumArgs)
            def doFit(spectrum, guess, **constantData):
                if len(constantData) > 0:
                    ff = self.fitfunc(**constantData)
                else:
                    ff = self.fitfunc
                if basinhopping:
                    result = scipy.optimize.basinhopping(ff, guess, minimizer_kwargs={'args': (spectrum,),
                                                                                      'bounds': self.limVals,
                                                                                      'method': method,
                                                                                      'jac': needJac})
                else:
                    result = scipy.optimize.minimize(ff, guess, args=(spectrum,), bounds=self.limVals, method=method, jac=needJac)
                #if result.success is False:
                #    raise DidNotConverge(result.message)
                return result.x, result.fun
        self.doFit = doFit #should return optimized parameters and residual as tuple
