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


import sys

class ProgressBar(object):
    def __init__(self, width=40):
        self.width = width
        self.pos = None
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\r[") 
        self.updateProgress(0)
    def updateProgress(self, value):
        pos = int(self.width * value)
        if self.pos == pos:
            return
        self.pos = pos
        sys.stdout.write(('-'*self.pos)+('\r['*self.pos))
        sys.stdout.flush()
    def done(self):
        sys.stdout.write("\n")

