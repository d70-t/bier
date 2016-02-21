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


import os
import datetime
import numpy as np

from zmqfitter import ZmqFitter
from celeryfitter import CeleryFitter
from globalfitter import GlobalFitter

from downwelling import DownwellingModel
from atmospheredata import AtmosphereData
from progressbar import ProgressBar
from wasienvi import WasiEnvi
from batchhelpers import fillModelConstantsFromSourceAndTaskDef

from runmacs.spec.util.sensorparameters import sensorParameters

fitters = {'ZmqFitter': ZmqFitter,
           'CeleryFitter': CeleryFitter,
           'GlobalFitter': GlobalFitter}

if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Runs WASI fit on all sky images")
    parser.add_argument("definitionfile", type=str, help="File with fit task definition")
    parser.add_argument("outname", type=str, help="Output file name")

    args = parser.parse_args()

    taskBasedir = os.path.dirname(args.definitionfile)

    print "loading task %s"%(args.definitionfile,)

    taskDef = yaml.load(open(args.definitionfile))

    source = WasiEnvi(taskDef['input'])
    wvlns = source.wvlns
    if wvlns is None:
        wvlns = sensorParameters['vnir']['loadWavelength']()
    wvlns = np.array(wvlns)
    spectrumSlice = slice(None)

    data = source.data[...,spectrumSlice]
    wvlns = wvlns[spectrumSlice]
    m = DownwellingModel(AtmosphereData(wvlns=wvlns))
    constantFields = fillModelConstantsFromSourceAndTaskDef(m, source, taskDef)

    outname = args.outname
    if outname[-4:] == '.npy':
        outname = outname[:-4]

    try:
        weightFile = taskDef['weights']
    except KeyError:
        weights = None
    else:
        weights= np.loadtxt(os.path.join(taskBasedir, weightFile), skiprows=2)
        weights = np.interp(wvlns, weights[:,0], weights[:,1])

    fitterName = taskDef.get('fitter', 'ZmqFitter')
    fitMethod  = taskDef.get('fitMethod', 'tnc')
    fitMinimumElevation = np.deg2rad(taskDef.get('fitMinimumElevation', 10))
    fitMinimumSCA = np.deg2rad(taskDef.get('fitMinimumSCA', 15))
    weightStrategy = taskDef.get('weightStrategy', 'irradiance')
    print 'using fitter: %s with %s-method'%(fitterName, fitMethod)

    if fitterName == 'GlobalFitter':
        #calculate spherical volume elements and skip horizon for the weights
        volumeElement = np.cos(constantFields['el'])
        spatialWeightMask = constantFields['el'] >= fitMinimumElevation
        sunCutMask = constantFields['sca'] >= fitMinimumSCA
        if weightStrategy == 'radiance':
            spatialWeight = volumeElement * spatialWeightMask * sunCutMask
        elif weightStrategy == 'irradiance':
            #sin(el) == cos(sza)
            spatialWeight = volumeElement * spatialWeightMask * sunCutMask * np.sin(constantFields['el'])
        else:
            raise ValueError('weight strategy "%s" is not defined!'%weightStrategy)
        print 'using "%s" weight strategy'%weightStrategy
        if weights is None:
            weights = (volumeElement * spatialWeight)[...,np.newaxis]
        else:
            weights = (volumeElement * spatialWeight)[...,np.newaxis] * weights[np.newaxis, np.newaxis, ...]

    print "weights shape:", weights.shape
    fitter = fitters[fitterName](data, wvlns, m, taskDef['guess'], taskDef['lims'], weights, method=fitMethod)

    start = datetime.datetime.now()
    print 'start @:', start
    progressBar = ProgressBar(50)
    result, residual = fitter.run(useNeighbors=False,progressBar=progressBar,constantData=constantFields)
    end = datetime.datetime.now()
    progressBar.done()
    print 'end @:', end

    np.save(outname+'.npy',result)
    np.save(outname+'.residual.npy',residual)

    outProps = {'taskDef': taskDef,
                'freeParams': fitter.paramNames,
                'fittingTime': (end-start).total_seconds()}
    with open(outname+'.prop.yaml', 'w') as f:
        yaml.dump(outProps, f)
    print 'data ist saved'
    print 'result variables are:', fitter.paramNames
    print 'duration was:', (end-start)

