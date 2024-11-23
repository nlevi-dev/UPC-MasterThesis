import os
import numpy as np
import datetime

class DataImputer:
    def __init__(self, path='data', debug=True, out='console', clear_log=True):
        self.path = path
        self.debug = debug
        self.out = out
        if out != 'console' and (not os.path.exists(out) or clear_log):
            open(out,'w').close()

    def log(self, msg):
        o = '{}| main [DATAIMPUTER] {}'.format(str(datetime.datetime.now())[11:16],msg)
        if self.out == 'console':
            print(o)
        else:
            with open(self.out,'a') as log:
                log.write(o+'\n')

    def readRaw(self):
        raw = np.genfromtxt(self.path+'/clinical.csv', delimiter=',', dtype=str)
        names = raw[:,1:]
        features = raw[1:,:]
        data = np.array(np.where(raw[1:,1:] == '',np.nan,raw[1:,1:]),np.float16)
        print(data)