import os
import re
import threading
import multiprocessing
import numpy as np
import datetime
from DataPoint import DataPoint
import LayeredArray as la
from util import *
from visual import showRadiomicsDist

np.seterr(invalid='ignore')
np.seterr(divide='raise')

def wrapperPreprocess(d):
    return d.preprocess()

def wrapperRadiomicsVoxel(d):
    d, f, k, b, r = d
    d.radiomicsVoxel(f,kernelWidth=k,binWidth=b,recompute=r)

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
    def __init__(self, path='data', debug=True, out='console', cores=None, partial=None, visualize=False):
        self.path = path
        self.debug = debug
        self.out = out
        self.visualize = visualize
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
        datapoints = [DataPoint(n,self.path,self.debug,self.out,self.visualize) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            shapes = pool.map(wrapperPreprocess, datapoints)
        shapes = np.array(shapes,np.uint16)
        np.save(self.path+'/preprocessed/shapes', shapes)
        self.log('Done preprocessing!')

    def radiomicsVoxel(self, kernelWidth=5, binWidth=25, recompute=True):
        # 4  (2/2)  [ 5, 49, 3,13,27s, 54s] 71  => 18
        # 6  (4/2)  [ 5, 71, 4,18,33s, 82s] 100 => 17
        # 6  (5/1)  [ 5, 65, 4,16,33s,100s] 92  => 15
        # 8  (5/3)  [ 6, 88, 5,22,45s,100s] 123 => 15
        # 12 (10/2) [ 8,147, 9,40,63s,145s] 207 => 17
        # 16 (11/5) [16,186,10,49,85s,187s] 266 => 17
        feature_classes = np.array(['firstorder','glcm','glszm','glrlm','ngtdm','gldm'])
        features = computeRadiomicsFeatureNames(feature_classes)
        np.save(self.path+'/preprocessed/features_vox',features)
        del features
        names = np.load(self.path+'/preprocessed/names.npy')
        names = self.partial(names)
        self.log('Started computing voxel based radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))

        global queue
        queue = []
        for n in names:
            for f in feature_classes:
                queue.append([DataPoint(n,self.path,self.debug,self.out,self.visualize),f,kernelWidth,binWidth,recompute])
        
        c1 = (5*self.cores)//6
        c2 = self.cores-c1
        threads = []
        threads += [threading.Thread(target=consumerThread,name='t'+str(i),args=[['glcm']]) for i in range(c1)]
        threads += [threading.Thread(target=consumerThread,name='t'+str(i),args=[['firstorder','glszm','glrlm','ngtdm','gldm']]) for i in range(c2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.log('Done computing feature classes!')
        self.log('Started concatenating feature classes!')
        for n in names:
            DataPoint(n,self.path,self.debug,self.out,self.visualize).radiomicsVoxelConcat(feature_classes, kernelWidth, binWidth)
        self.log('Done computing voxel based radiomic features!')

    def deletePartialData(self, kernelWidth=5, binWidth=25):
        self.log('Started deleting partial data!')
        feature_classes = np.array(['firstorder','glcm','glszm','glrlm','ngtdm','gldm'])
        names = np.load(self.path+'/preprocessed/names.npy')
        for n in names:
            for f in feature_classes:
                p = '{}/preprocessed/{}/t1_radiomics_raw_k{}_b{}_{}.npy'.format(self.path,n,kernelWidth,binWidth,f)
                if os.path.exists(p):
                    os.remove(p)
        self.log('Done deleting partial data!')

    def radiomics(self, binWidth=25):
        features = computeRadiomicsFeatureNames(['firstorder','glcm','glszm','glrlm','ngtdm','gldm','shape'])
        np.save(self.path+'/preprocessed/features',features)
        del features
        names = np.load(self.path+'/preprocessed/names.npy')
        names = self.partial(names)
        self.log('Started computing radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out,self.visualize) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            pool.map(wrapperRadiomics, [[d,binWidth] for d in datapoints])
        self.log('Done computing radiomic features!')
    
    def scaleRadiomics(self, kernelWidth=5, binWidth=25):
        names = np.load(self.path+'/preprocessed/names.npy')

        self.log('Started computing scale factors for radiomics!')
        features = np.load(self.path+'/preprocessed/features.npy')
        mi = np.repeat(np.array([sys.maxsize],np.float32),len(features))
        ma = np.repeat(np.array([-sys.maxsize],np.float32),len(features))
        for n in names:
            for a in ['t1_mask','roi','targets']:
                arr = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_b{}_{}.npy'.format(n,binWidth,a))
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, 0)
                mi = np.min(np.concatenate([np.expand_dims(mi,0),np.expand_dims(np.min(arr,0),0)],0),0)
                ma = np.max(np.concatenate([np.expand_dims(ma,0),np.expand_dims(np.max(arr,0),0)],0),0)
        factors = np.concatenate([np.expand_dims(mi,-1),np.expand_dims(ma,-1)],-1)
        np.save(self.path+'/preprocessed/features_scale_b{}'.format(binWidth), factors)
        del mi; del ma; del factors
        self.log('Done computing scale factors for radiomics!')

        self.log('Started computing scale factors for voxel based radiomics!')
        features_vox = np.load(self.path+'/preprocessed/features_vox.npy')
        shape = np.max(np.load(self.path+'/preprocessed/shapes.npy'),0)
        factors_vox = []
        distributions = []
        for i in range(len(features_vox)):
            self.log('Started feature {}!'.format(features_vox[i]))
            con = np.zeros((len(names),)+tuple(shape),np.float32)
            for j in range(len(names)):
                raw = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_k{}_b{}.npy'.format(names[j],kernelWidth,binWidth),mmap_mode='r')
                raw = raw[:,:,:,i]
                center = (shape-np.array(raw.shape))//2
                con[j,center[0]:center[0]+raw.shape[0],
                    center[1]:center[1]+raw.shape[1],
                    center[2]:center[2]+raw.shape[2]] = raw
            f, dis = scaleRadiomics(con)
            if self.visualize:
                showRadiomicsDist(features_vox[i],dis[0:2],dis[2:4],f[2]=='log10')
            factors_vox.append(f)
            distributions.append(dis)
            self.log('Done feature {}!'.format(features_vox[i]))
        np.save(self.path+'/preprocessed/features_scale_vox_distributions_k{}_b{}'.format(kernelWidth,binWidth), np.array(distributions))
        np.save(self.path+'/preprocessed/features_scale_vox_k{}_b{}'.format(kernelWidth,binWidth), np.array(factors_vox))
        self.log('Done computing scale factors for voxel based radiomics!')

    def preloadData(self, kernelWidth=5, binWidth=25):
        names = np.load(self.path+'/preprocessed/names.npy')

        self.log('Started preloading data!')
        factors = np.load(self.path+'/preprocessed/features_scale_b{}.npy'.format(binWidth))
        factors_vox = np.load(self.path+'/preprocessed/features_scale_vox_k{}_b{}.npy'.format(kernelWidth,binWidth))
        for i in range(len(names)):
            name = names[i]
            self.log('Started preloading {}!'.format(name))
            raw = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_k{}_b{}.npy'.format(name,kernelWidth,binWidth))
            mask = la.load(self.path+'/preprocessed/{}/roi.pkl'.format(name))
            mask_left = mask[:,:,:,0].flatten()
            mask_right = mask[:,:,:,1].flatten()
            res = np.zeros(raw.shape,np.float16)
            res_flat_left = np.zeros((np.count_nonzero(mask_left),len(factors_vox)),np.float16)
            res_flat_right = np.zeros((np.count_nonzero(mask_right),len(factors_vox)),np.float16)
            for j in range(len(factors_vox)):
                slc = raw[:,:,:,j]
                if factors_vox[j][2] == 'log10':
                    slc = np.log10(slc+1)
                    fac = np.array(factors_vox[j][3:5],np.float32)
                else:
                    fac = np.array(factors_vox[j][0:2],np.float32)
                slc = np.array((slc-fac[0])/(fac[1]-fac[0]),np.float16)
                res[:,:,:,j] = slc
                flat = slc.flatten()
                res_flat_left[:,j] = flat[mask_left]
                res_flat_right[:,j] = flat[mask_right]
            con = la.load(self.path+'/preprocessed/{}/connectivity.pkl'.format(name))
            con_flat_left = np.zeros((np.count_nonzero(mask_left),con.shape[-1]),np.float16)
            con_flat_right = np.zeros((np.count_nonzero(mask_right),con.shape[-1]),np.float16)
            for j in range(con.shape[-1]):
                slc = con[:,:,:,j].flatten()
                con_flat_left[:,j] = slc[mask_left]
                con_flat_right[:,j] = slc[mask_right]
            tar = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_b{}_targets.npy'.format(name,binWidth))
            roi = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_b{}_roi.npy'.format(name,binWidth))
            bra = np.load(self.path+'/preprocessed/{}/t1_radiomics_raw_b{}_t1_mask.npy'.format(name,binWidth))
            mi = np.expand_dims(factors[:,0],0)
            ma = np.expand_dims(factors[:,1],0)
            bra = np.expand_dims(bra,0)
            tar = np.array((tar-np.repeat(mi,len(tar),0))/np.repeat((ma-mi),len(tar),0),np.float16)
            roi = np.array((roi-np.repeat(mi,len(roi),0))/np.repeat((ma-mi),len(roi),0),np.float16)
            bra = np.array((bra-np.repeat(mi,len(bra),0))/np.repeat((ma-mi),len(bra),0),np.float16)
            self.log('Saving {}!'.format(name))
            if not os.path.isdir(self.path+'/preloaded/'+name):
                self.log('Creating output directory at \'{}\'!'.format(self.path+'/preloaded/'+name))
                os.makedirs(self.path+'/preloaded/'+name,exist_ok=True)
            np.save(self.path+'/preloaded/{}/t1_radiomics_norm_k{}_b{}.npy'.format(name,kernelWidth,binWidth),res)
            np.save(self.path+'/preloaded/{}/t1_radiomics_norm_left_k{}_b{}.npy'.format(name,kernelWidth,binWidth),res_flat_left)
            np.save(self.path+'/preloaded/{}/t1_radiomics_norm_right_k{}_b{}.npy'.format(name,kernelWidth,binWidth),res_flat_right)
            np.save(self.path+'/preloaded/{}/connectivity_left.npy'.format(name),con_flat_left)
            np.save(self.path+'/preloaded/{}/connectivity_right.npy'.format(name),con_flat_right)
            np.save(self.path+'/preloaded/{}/t1_radiomics_scale_b{}_targets.npy'.format(name,binWidth),tar)
            np.save(self.path+'/preloaded/{}/t1_radiomics_scale_b{}_roi.npy'.format(name,binWidth),roi)
            np.save(self.path+'/preloaded/{}/t1_radiomics_scale_b{}_t1_mask.npy'.format(name,binWidth),bra)
            self.log('Done preloading {}!'.format(name))
        self.log('Done preloading data!')

