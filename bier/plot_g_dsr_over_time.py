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
import pylab as plt
import datetime
import re
import yaml
import collections

if __name__ == '__main__':
    from glob import glob
    import os
    files = glob(os.path.join(os.path.dirname(__file__), 'results', 'rayleigh_only_munich*.auto.npy'))
    time = collections.defaultdict(list)
    mean = collections.defaultdict(list)
    median = collections.defaultdict(list)
    dateRe = re.compile('([0-9]{4})_([0-9]{2})_([0-9]{2})_([0-9]{2})')
    for fn in sorted(files):
        m = dateRe.search(fn)
        if m is None:
            continue
        #date = datetime.datetime(*map(int, [m.group(i) for i in range(1,5)]))
        meta = yaml.load(open(fn.replace('.npy', '.prop.yaml')))
        try:
            i_g_dsr = meta['freeParams'].index('g_dsr')
        except ValueError:
            continue
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        hour = int(m.group(4))
        key = '%04d_%02d_%02d'%(year, month, day)
        time[key].append(hour)
        d = np.load(fn)[...,i_g_dsr]
        mean[key].append(np.mean(d))
        median[key].append(np.median(d))
    fig, axes = plt.subplots(2)
    axes[0].set_title('g_dsr mean over time')
    axes[1].set_title('g_dsr median over time')
    for k,t in sorted(time.items()):
        axes[0].plot(t, mean[k], label=k)
        axes[1].plot(t, median[k], label=k)
    axes[0].legend(loc=0)
    axes[1].legend(loc=0)
    plt.show()
