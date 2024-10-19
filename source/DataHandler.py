import os
import re
import threading
import multiprocessing
import numpy as np
import datetime
from DataPoint import DataPoint
from util import computeRadiomicsFeatureNames

np.seterr(invalid='ignore')

def wrapperPreprocess(d):
    return d.preprocess()

def wrapperRadiomicsVoxel(d):
    d, f, k, b, r = d
    d.radiomicsVoxel(f,kernelWidth=k,binWidth=b,recompute=r)

def wrapperRadiomicsVoxelConcat(d):
    d, f, k, b = d
    d.radiomicsVoxelConcat(f,kernelWidth=k,binWidth=b)

lock = threading.Lock()
queue = []

def consumerThread(pref):
    global queue
    with multiprocessing.Pool(1) as t:
        while True:
            with lock:
                if len(queue) == 0:
                    return
                idx = 0
                while True:
                    if idx >= len(queue):
                        d = queue.pop(0)
                        break
                    if queue[idx][1] in pref:
                        d = queue.pop(idx)
                        break
                    else:
                        idx += 1
            t.map(wrapperRadiomicsVoxel,[d])

def wrapperRadiomics(d):
    d, b = d
    return d.radiomics(binWidth=b)

class DataHandler:
    def __init__(self, path='data', debug=False, out='console', cores=None, partial=None):
        self.path = path
        self.debug = debug
        self.out = out
        self.partial = None
        if isinstance(partial, tuple):
            if len(partial) == 2:
                if isinstance(partial[0], float) or isinstance(partial[1], float):
                    self.partial = lambda n:n[range(int(len(n)*partial[0]),int(len(n)*partial[1]))]
                else:
                    self.partial = lambda n:n[range(partial[0],partial[1])]
        elif isinstance(partial, list):
            self.partial = lambda n:n.take(partial)
        elif isinstance(partial, range):
            self.partial = lambda n:n[partial]
        elif callable(partial):
            self.partial = partial
        if self.partial is None:
            self.partial = lambda n:n
        maxcores = multiprocessing.cpu_count()
        if callable(cores):
            self.cores = cores(maxcores)
        elif isinstance(cores, int):
            self.cores = cores
        elif isinstance(cores, float):
            self.cores = int(maxcores*cores)
        else:
            self.cores = maxcores
        if self.cores == 0:
            self.cores = 1
        if self.cores > maxcores or self.cores < 0:
            self.cores = maxcores
        if out != 'console':
            open(out,'w').close()

    def log(self, msg):
        o = '{}| main [DATAHANDLER] {}'.format(str(datetime.datetime.now())[11:16],msg)
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
        np.save(self.path+'/preprocessed/names', names)
        names = self.partial(names)

        labels = np.array(['limbic','executive','rostral-motor','caudal-motor','parietal','occipital','temporal'])
        np.save(self.path+'/preprocessed/labels', labels)

        self.log('Starting preprocessing {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            shapes = pool.map(wrapperPreprocess, datapoints)
        shapes = np.array(shapes,np.uint16)
        np.save(self.path+'/preprocessed/shapes', shapes)
        self.log('Done preprocessing!')

    def radiomicsVoxel(self, kernelWidth=5, binWidth=25, recompute=True):
        feature_classes = np.array(['firstorder','glcm','glszm','glrlm','ngtdm','gldm'])
        features = computeRadiomicsFeatureNames(feature_classes)
        np.save(self.path+'/preprocessed/features_vox',features)
        del features
        names = np.load(self.path+'/preprocessed/names.npy')
        names = self.partial(names)
        self.log('Starting computing voxel based radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))

        global queue
        queue = []
        for n in names:
            for f in feature_classes:
                queue.append([DataPoint(n,self.path,self.debug,self.out),f,kernelWidth,binWidth,recompute])
        
        c1 = (2*self.cores)//3
        c2 = self.cores-c1
        threads = []
        threads += [threading.Thread(target=consumerThread,name='t'+str(i),args=[['glcm']]) for i in range(c1)]
        threads += [threading.Thread(target=consumerThread,name='t'+str(i),args=[['firstorder','glszm','glrlm','ngtdm','gldm']]) for i in range(c2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log('Done computing feature classes!')
        self.log('Concatenating feature classes!')
        with multiprocessing.Pool(self.cores) as pool:
            pool.map(wrapperRadiomicsVoxelConcat, [[DataPoint(n,self.path,self.debug,self.out),feature_classes,kernelWidth,binWidth] for n in names])
        self.log('Done computing voxel based radiomic features!')

    def radiomics(self, binWidth=25):
        features = computeRadiomicsFeatureNames(['firstorder','glcm','glszm','glrlm','ngtdm','gldm','shape'])
        np.save(self.path+'/preprocessed/features',features)
        del features
        names = np.load(self.path+'/preprocessed/names.npy')
        names = self.partial(names)
        self.log('Starting computing radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            pool.map(wrapperRadiomics, [[d,binWidth] for d in datapoints])
        self.log('Done computing radiomic features!')
