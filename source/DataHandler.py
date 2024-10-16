import os
import re
import multiprocessing
import numpy as np
from DataPoint import DataPoint

def wrapperPreprocess(d):
    return d.preprocess()

class DataHandler:
    def __init__(self, path='data', debug=False, out='console', cores=None):
        self.path = path
        self.debug = debug
        self.out = out
        maxcores = multiprocessing.cpu_count()
        if callable(cores):
            self.cores = cores(maxcores)
        elif isinstance(cores, int):
            self.cores = cores
        else:
            self.cores =maxcores
        if self.cores < 1:
            self.cores = 1
        if self.cores > maxcores:
            self.cores = maxcores
        if out != 'console':
            open(out,'w').close()
    
    def log(self, msg):
        o = 'main [DATAHANDLER] {}'.format(msg)
        if self.out == 'console':
            print(o)
        else:
            with open(self.out,'a') as log:
                log.write(o+'\n')

    def preprocess(self):
        names = os.listdir(self.path+'/raw')
        r = re.compile('[CH]\d.*')
        names = [s for s in names if r.match(s)]
        blacklist = ['H24_1','H27_1','H29_1']
        names = [n for n in names if n not in blacklist]
        names = sorted(names)
        np.save('data/preprocessed/names', names)

        labels = np.array(['limbic','executive','rostral-motor','caudal-motor','parietal','occipital','temporal'])
        np.save('data/preprocessed/labels', labels)

        self.log('Starting preprocessing {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            shapes = pool.map(wrapperPreprocess, datapoints)
        shapes = np.array(shapes,np.uint16)
        np.save('data/preprocessed/shapes', shapes)
        self.log('Done preprocessing!')