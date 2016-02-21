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


from fitter import PointwiseFieldModelFitter, LocalFitterMixin, DidNotConverge
import zmq
import numpy as np
import multiprocessing
import time
import json

worksocket = 'ipc:///tmp/wasifit_toWorkers'
ressocket = 'ipc:///tmp/wasifit_fromWorkers'
#worksocket = 'inproc://toWorkers'
#ressocket = 'inproc://fromWorkers'

class ZmqFitter(PointwiseFieldModelFitter, LocalFitterMixin):
    def _postInit(self):
        self._localPostInit()
        self.profileName = None
        self.poolsize = 8
    
    def _prepareRun(self):
        ctx=zmq.Context()
        self.toWorkers=ctx.socket(zmq.PUSH)
        self.fromWorkers=ctx.socket(zmq.PULL)
        self.toWorkers.bind(worksocket)
        self.fromWorkers.bind(ressocket)
        
        self.workers = []
        for i in xrange(self.poolsize):
            if self.profileName is None:
                w = multiprocessing.Process(target=self.worker)
            else:
                w = multiprocessing.Process(target=self.profilingWorker,args=(self.profileName,i))
            w.start()
            self.workers.append(w)
        time.sleep(1)
    def _cleanupRun(self):
        for _ in range(self.poolsize * 10):
            self.toWorkers.send_multipart(['exit', 'done'])
        for w in self.workers:
            w.join()
    def _queuePoint(self, point, guess, constantData):
        self.toWorkers.send_multipart(['data', json.dumps({'point': point, 'guess': guess, 'constantData': constantData})])
    def _receivePoints(self):
        header, data = self.fromWorkers.recv_multipart()
        if header == 'ok':
            data = json.loads(data)
            point = tuple(data['point'])
            self._markSolved(point, data['res']['fitValues'], data['res']['residual'])
        elif header == 'error':
            data = json.loads(data)
            point = tuple(data['point'])
            self._markFailed(point, data['reason'])
    def profilingWorker(self,profileName,i):
        import cProfile
        cProfile.runctx('self.worker()', globals(), locals(), '%s-%s.prof'%(profileName, i))
    def worker(self):
        ctx=zmq.Context()
        recv=ctx.socket(zmq.PULL)
        send=ctx.socket(zmq.PUSH)
        recv.connect(worksocket)
        send.connect(ressocket)
        
        while True:
            #print 'waiting for message'
            msg = recv.recv_multipart()
            #print 'got:', msg
            if msg[0] == 'exit':
                break
            if msg[0] == 'data':
                req = json.loads(msg[1])
                spectrum = self.field[tuple(req['point'])]
                guess = req['guess']
                constantData = req['constantData']
                try:
                    popt, residual = self.doFit(spectrum, guess, **constantData)
                    send.send_multipart(['ok', json.dumps({'point': req['point'], 'res':{'fitValues': popt.tolist(), 'residual': float(residual)}})])
                except DidNotConverge as e:
                    send.send_multipart(['error', json.dumps({'point': req['point'], 'reason': e.reason})])
