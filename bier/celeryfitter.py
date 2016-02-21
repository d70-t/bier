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


from fitter import FieldModelFitter
import scipy.optimize
import wasicelerytasks
import celery.result
import numpy as np
import itertools
import time

class CeleryFitter(FieldModelFitter):
    def _postInit(self):
        self.fixedValues = self.model.fixedValues
        self.paramNames = self.model.getFreeParamNames()
        self.constantNames = self.model.getConstantParamNames()

    def _prepareRun(self):
        self.resultSet = celery.result.ResultSet([])
        self.openResults = []
    def _receivePoints(self):
        if len(self.openPoints) == 0:
            print '---------------'
            if len(self.openResults) < 10 and len(self.openPoints) == 0 or len(self.nextPoints) == 0:
                for res in self.openResults:
                    print res.id, res.state
                #maybe some hack is needed here
        anythingChanged = False
        while not anythingChanged:
            for res in self.openResults[:]:
                if res.ready():
                    self.completeCallback(res.result)
                    anythingChanged = True
            if not anythingChanged:
                time.sleep(.01)

    def _queuePoint(self, point, guess, constantData):
        futureResult = wasicelerytasks.doFit.delay(point=point,
                                                   spectrum=self.field[point],
                                                   guess=guess,
                                                   wvlns=self.wvlns,
                                                   fixedValues=self.fixedValues,
                                                   method=self.method,
                                                   weights=self.weights,
                                                   limVals=self.limVals,
                                                   constantData=constantData)
        self.openResults.append(futureResult)
    def completeCallback(self, value):
        #print 'got', value
        header = value[0]
        if header == 'ok':
            point, fitValues = value[1]
            try:
                self.openResults.remove(point)
            except ValueError:
                pass #print 'could not remove point', point
            self._markSolved(point, fitValues)
        if header == 'error':
            point = value[1]
            try:
                self.openResults.remove(point)
            except ValueError:
                pass #print 'could not remove point', point
            self._markFailed(point)

