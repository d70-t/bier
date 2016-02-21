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
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import scipy.optimize
from scipy.ndimage.morphology import binary_erosion

from downwelling import DownwellingModel
from atmospheredata import AtmosphereData
from wasienvi import WasiEnvi
from batchhelpers import fillModelConstantsFromSourceAndTaskDef
from rayleigh import verticalRayleighCoefficient, sca2rayleigh

from runmacs.spec.util.specrend import SpecRGBConverter, postProcessRGB

import pint

colorLims = {'H_oz': (0, 1),
             'WV': (0, 2),
             'RH': (0, 1),
             'alpha': (0, 3),
             'beta': (0, 2),
             'g_dsa': (0, .1),
             'g_dsr': (0, .5),
             'residual': (0, 25),
             }
             
figure_args = {'figsize': (12,8),
               }

def cutMean(a):
    return a[...,168:747].mean(axis=-1) #sigma(g1) <= 5%
def percentDifference(a,b):
    return 100. * cutMean(a-b)/cutMean(a)

if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Creates some previews for WASI fit results")
    parser.add_argument("resultname", type=str, help="Result file name")

    args = parser.parse_args()

    ureg = pint.UnitRegistry()
    targetRadianceUnits = ureg('mW/(m^2 nm sr)')

    resultName = args.resultname
    if resultName[-4:] == '.npy':
        resultName = resultName[:-4]
    #resultBaseDir, resultBaseName  = os.path.split(resultName)

    meta = yaml.load(open(resultName + '.prop.yaml'))
    result = np.load(resultName + '.npy')
    residual = np.load(resultName + '.residual.npy')
    if len(result.shape) == 1:
        fitGlobal = True
    else:
        fitGlobal = False

    source = WasiEnvi(meta['taskDef']['input'], ureg)
    print u"source units: {:~P}".format(source.units)
    unitConversionFactor = source.units.to(targetRadianceUnits).magnitude
    print u"conversion factor:", unitConversionFactor
    extent = np.rad2deg(np.array([source.grid[0,0,0], source.grid[0,0,-1], source.grid[1,0,0], source.grid[1,-1,0]]))
    wvlns = np.array(source.wvlns)

    rend = SpecRGBConverter(wvlns)
    rgbSource = rend.spectrum_to_rgb(source.data, postprocess=False)

    m = DownwellingModel(AtmosphereData(wvlns=wvlns))
    constantFields = fillModelConstantsFromSourceAndTaskDef(m, source, meta['taskDef'])
    if fitGlobal:
        freeParams, modelFunction = m.getCompiledE(fitGlobal=True)
    else:
        freeParams, modelFunction = m.getCompiledE(useFields=True)
    modelParams = []
    for name in freeParams:
        if name in constantFields:
            modelParams.append(constantFields[name][...,np.newaxis])
        else:
            if fitGlobal:
                modelParams.append(result[...,meta['freeParams'].index(name)])
            else:
                modelParams.append(result[...,meta['freeParams'].index(name),np.newaxis])
    print result.shape, freeParams, constantFields.keys()
    reconstructedImage = modelFunction(*modelParams)
    rgbReconstructed = rend.spectrum_to_rgb(reconstructedImage, postprocess=False)
    
    rgbDifference = rgbSource - rgbReconstructed
    
    rgbSource, rgbReconstructed = postProcessRGB(np.array((
        rgbSource, rgbReconstructed)))
    rgbDifference = postProcessRGB(rgbDifference)
    
    stdResidual = residual**.5 * unitConversionFactor

    print "stdResidual[0]:", stdResidual.ravel()[0]

    pp = PdfPages(resultName + '.pdf')

    try:
        #create fit parameter images
        subplotCount = len(meta['freeParams'])
        if subplotCount > 0 and not fitGlobal:
            if subplotCount < 3:
                fig, axes = plt.subplots(subplotCount, **figure_args)
                if subplotCount == 1:
                    axes = [axes]
            else:
                fig, axes = plt.subplots((subplotCount + 1)/2,2, **figure_args)
                axes = axes.ravel()
            for ax, label, res in zip(axes, meta['freeParams'], np.rollaxis(result,2)):
                ax.set_title(label + '; mean: %.3f; median: %.3f'%(res.mean(), np.median(res)))
                ax.set_xlabel('azimuth [deg]')
                ax.set_ylabel('elevation [deg]')
                if label in colorLims:
                    img = ax.imshow(res[::-1], extent=extent, vmin=colorLims[label][0], vmax=colorLims[label][1])
                else:
                    img = ax.imshow(res[::-1], extent=extent)
                fig.colorbar(img, ax=ax)
            fig.tight_layout()
            pp.savefig(fig)

        if not fitGlobal:
            fig, ax = plt.subplots(1, **figure_args)
            ax.set_title(u'standardized residual [${:~L}$]'.format(targetRadianceUnits))
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
            img = ax.imshow(stdResidual[::-1],extent=extent, vmin=colorLims['residual'][0], vmax=colorLims['residual'][1])
            fig.colorbar(img, ax=ax)
            fig.tight_layout()
            pp.savefig(fig)

        #rayleigh parametrization analysis
        if 'g_dsr' in meta['freeParams'] and False:
            if 'beta' in meta['freeParams']:
                fitMask = result[...,meta['freeParams'].index('beta')] < .5
                fitMask = binary_erosion(fitMask, iterations=3)
                restrictData = lambda x: x[fitMask]
            else:
                restrictData = lambda x: x.ravel()
            fR = 1 + np.cos(source.sca)**2
            normalized_g_dsr = restrictData(result[...,meta['freeParams'].index('g_dsr')] / fR)
            sourceEl = restrictData(source.el)
            def rayleighFitFun(el, e, t):
                return e * verticalRayleighCoefficient(el, t)
            popt, pcov = scipy.optimize.curve_fit(rayleighFitFun, sourceEl, normalized_g_dsr, (0.01,0.2))

            g_dsr_computed = sca2rayleigh(source.sca, source.el, *popt)

            fig, ax = plt.subplots(3, **figure_args)
            ax, ax2, ax3 = ax
            ax.set_title(u'verticalRayleighCoefficient (%e,%e)'%tuple(popt))
            ax.set_xlabel('elevation [deg]')
            ax.set_ylabel('value [a.u.]')
            ax.scatter(np.rad2deg(sourceEl), normalized_g_dsr)
            elev = np.arange(0,90)
            ax.plot(elev, rayleighFitFun(np.deg2rad(elev), *popt), color='red')
            ax = ax2
            ax.set_title(u'computed g_dsr')
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
            img = ax.imshow(g_dsr_computed[::-1],extent=extent, vmin=colorLims['g_dsr'][0], vmax=colorLims['g_dsr'][1])
            fig.colorbar(img, ax=ax)
            ax = ax3
            ax.set_title(u'fit - computed')
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
            img = ax.imshow((result[...,meta['freeParams'].index('g_dsr')] - g_dsr_computed)[::-1],extent=extent, vmin=-colorLims['g_dsr'][1]/5., vmax=colorLims['g_dsr'][1]/5.)
            fig.colorbar(img, ax=ax)
            fig.tight_layout()
            pp.savefig(fig)


        #create rgb images
        fig, axes = plt.subplots(2,2, **figure_args)
        for ax, title, data, lims in zip(axes.ravel(),
                                   ('original image', 'reconstructed image', 'color difference', 'averaged difference'),
                                   (rgbSource, rgbReconstructed, rgbDifference, percentDifference(reconstructedImage, source.data)),
                                   (None, None, None, (-10,10))):
            ax.set_title(title)
            ax.set_xlabel('azimuth [deg]')
            ax.set_ylabel('elevation [deg]')
            if lims is None:
                ax.imshow(data[::-1],extent=extent)
            else:
                try:
                    vmin, vmax = lims
                except:
                    img = ax.imshow(data[::-1],extent=extent)
                else:
                    img = ax.imshow(data[::-1],extent=extent, vmin=vmin, vmax=vmax)
                fig.colorbar(img, ax=ax, orientation="horizontal")
        fig.tight_layout()
        pp.savefig(fig)

    except:
        pp.close()
        os.unlink(resultName + '.pdf')
        raise
    else:
        pp.close()
