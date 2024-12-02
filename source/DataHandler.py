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

def wrapperNormalize(d):
    return d.normalize()
def wrapperRegister(d):
    return d.register()
def wrapperPreprocess(d):
    return d.preprocess()
def wrapperRadiomicsVoxel(d):
    d, f, k, b, r, a, i, da, ma, cu = d
    d.radiomicsVoxel(f,kernelWidth=k,binWidth=b,recompute=r,absolute=a,inp=i,data=da,mask=ma,cutout=cu)
def wrapperRadiomics(d):
    d, b, a, i = d
    return d.radiomics(binWidth=b,absolute=a,inp=i)

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

class DataHandler:
    def __init__(self, path='data', space='native', debug=True, out='console', cores=None, partial=None, visualize=False, clear_log=True):
        self.path = path
        self.space = space
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
        if out != 'console' and (not os.path.exists(out) or clear_log):
            open(out,'w').close()
        if not os.path.exists(self.path+'/preprocessed/names.npy'):
            names = os.listdir(self.path+'/raw')
            r = re.compile('[CH]\d.*')
            names = [s for s in names if r.match(s)]
            names = sorted(names)
            np.save(self.path+'/preprocessed/names', names)
        self.names = np.load(self.path+'/preprocessed/names.npy')
        self.names = self.partial(self.names)
        if os.path.exists(self.path+'/preprocessed/missing.pkl'):
            self.missing = pickleLoad(self.path+'/preprocessed/missing.pkl')
        if self.space == 'normalized':
            self.names = [n for n in self.names if n not in self.missing['normalized']]

    def log(self, msg):
        o = '{}| main [DATAHANDLER] {}'.format(str(datetime.datetime.now())[11:16],msg)
        if self.out == 'console':
            print(o)
        else:
            with open(self.out,'a') as log:
                log.write(o+'\n')
    
    def processClinical(self, imputeStrategy=[['CAP'],['TFC','UHDRSmotor','Digit_symbol_correct','stroop_word']]):
        raw = np.genfromtxt(self.path+'/clinical.csv', delimiter=',', dtype=str)
        features = raw[0,1:]
        names_clinical = list(raw[1:,0])
        data = np.array(np.where(raw[1:,1:] == '',np.nan,raw[1:,1:]),np.float32)
        mins = np.nanmin(data,0)
        maxs = np.nanmax(data,0)
        factors = np.concatenate([np.expand_dims(mins,-1),np.expand_dims(maxs,-1)],-1)
        mins = np.repeat(np.expand_dims(mins,0),len(data),0)
        maxs = np.repeat(np.expand_dims(maxs,0),len(data),0)
        tmp = np.where(np.isnan(data),0,data)
        tmp = (tmp-mins)/(maxs-mins)
        data = np.where(np.isnan(data),data,tmp)
        fromFeatures = []
        for i in range(len(imputeStrategy)):
            fromFeatures += imputeStrategy[i]
            fromIdxs = maskFromStrings(features,fromFeatures)
            if i == len(imputeStrategy)-1:
                toIdxs = None
            else:
                toIdxs = maskFromStrings(features,imputeStrategy[i+1])
            data = impute(data,fromIdxs,toIdxs)
        np.save(self.path+'/preprocessed/features_clinical',features)
        np.save(self.path+'/preprocessed/features_scale_clinical',factors)
        miss = [n for n in self.names if n not in names_clinical]
        self.missing['clinical'] = miss
        pickleSave(self.path+'/preprocessed/missing.pkl', self.missing)
        names = [n for n in self.names if n not in self.missing['clinical']]
        for n in names:
            idx = names_clinical.index(n)
            np.save(self.path+'/native/preloaded/'+n+'/clinical',data[idx,:])
            np.save(self.path+'/normalized/preloaded/'+n+'/clinical',data[idx,:])

    def register(self):
        self.log('Starting registering {} datapoints on {} core{}!'.format(len(self.names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out,self.visualize,create_folders=True) for n in self.names]
        with multiprocessing.Pool(self.cores) as pool:
            missing_raw = pool.map(wrapperRegister, datapoints)
            missing = {}
            for element in missing_raw:
                keys = list(element.keys())
                keys = [k for k in keys if k != 'name']
                for key in keys:
                    if not element[key]:
                        if key in missing.keys():
                            missing[key].append(element['name'])
                        else:
                            missing[key] = [element['name']]
        msg = 'MISSING:'
        for k in list(missing.keys()):
            msg += '\n                 '+k+': '+str(missing[k])
        self.log(msg)
        pickleSave(self.path+'/preprocessed/missing.pkl', missing)
        self.log('Done registering!')

    def normalize(self):
        self.log('Starting normalizing {} datapoints on {} core{}!'.format(len(self.names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path,self.debug,self.out,self.visualize) for n in self.names]
        with multiprocessing.Pool(self.cores) as pool:
            missing_raw = pool.map(wrapperNormalize, datapoints)
            missing = pickleLoad(self.path+'/preprocessed/missing.pkl')
            for element in missing_raw:
                if not element['normalized']:
                    if 'normalized' in missing.keys():
                        missing['normalized'].append(element['name'])
                    else:
                        missing['normalized'] = [element['name']]
        self.log('MISSING:\n                 normalized: '+str(missing['normalized']))
        pickleSave(self.path+'/preprocessed/missing.pkl', missing)
        self.log('Done normalizing!')

    def preprocess(self):
        labels = np.array(['limbic','executive','rostral-motor','caudal-motor','parietal','occipital','temporal'])
        np.save(self.path+'/preprocessed/labels', labels)

        self.log('Starting preprocessing {} datapoints on {} core{}!'.format(len(self.names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path+'/'+self.space,self.debug,self.out,self.visualize) for n in self.names]
        with multiprocessing.Pool(self.cores) as pool:
            shapes = pool.map(wrapperPreprocess, datapoints)
        shapes = np.array(shapes,np.uint16)
        np.save(self.path+'/'+self.space+'/preprocessed/shapes', shapes)
        self.log('Done preprocessing!')

    def radiomicsVoxel(self, kernelWidth=5, binWidth=25, recompute=True, absolute=True, inp='t1', fastOnly=False, basalOnly=True):
        feature_classes = np.array(['ngtdm','gldm'] if fastOnly else ['firstorder','glcm','glszm','glrlm','ngtdm','gldm'])
        features = computeRadiomicsFeatureNames(feature_classes)
        np.save(self.path+'/preprocessed/features_vox',features)
        del features
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        self.log('Started computing voxel based radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        if (not recompute) and os.path.exists('{}/{}/preprocessed/{}/{}_radiomics_raw_k{}_b{}.npy'.format(self.path,self.space,names[-1],inp,kernelWidth,binstr)):
            self.log('Already computed, skipping!')
            return
        global queue
        queue = []
        for n in names:
            if basalOnly:
                cutout = la.load(self.path+'/'+self.space+'/preprocessed/'+n+'/mask_basal.pkl')
                mask_basal = np.logical_or(cutout[:,:,:,0],cutout[:,:,:,1])
                bounds = findMaskBounds(mask_basal)
                cutout = cutout[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
                mask_basal = mask_basal[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
                mask_brain = np.load(self.path+'/'+self.space+'/preprocessed/'+n+'/mask_brain.npy')
                mask_brain = mask_brain[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
                mask = np.zeros(mask_brain.shape,np.bool_)
                data = np.load(self.path+'/'+self.space+'/preprocessed/'+n+'/'+inp+'.npy')
                data = data[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
                k = (kernelWidth-1)//2
                for x in range(mask_basal.shape[0]):
                    for y in range(mask_basal.shape[1]):
                        for z in range(mask_basal.shape[2]):
                            x1 = x-k
                            x2 = x+k+1
                            y1 = y-k
                            y2 = y+k+1
                            z1 = z-k
                            z2 = z+k+1
                            if x1 < 0: x1 = 0
                            if y1 < 0: y1 = 0
                            if z1 < 0: z1 = 0
                            if x2 > mask_basal.shape[0]: x2 = mask_basal.shape[0]
                            if y2 > mask_basal.shape[1]: y2 = mask_basal.shape[1]
                            if z2 > mask_basal.shape[2]: z2 = mask_basal.shape[2]
                            mask[x1:x2,y1:y2,z1:z2] = 1
                mask = np.logical_or(mask,mask_brain)
                del mask_basal
                del mask_brain
            else:
                data = None
                mask = None
                cutout = None
            for f in feature_classes:
                queue.append([DataPoint(n,self.path+'/'+self.space,self.debug,self.out,self.visualize),f,kernelWidth,binWidth,recompute,absolute,inp,data,mask,cutout])
        
        c2 = (3*self.cores)//8
        c1 = self.cores-c2
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
            DataPoint(n,self.path+'/'+self.space,self.debug,self.out,self.visualize).radiomicsVoxelConcat(feature_classes, kernelWidth, binWidth, absolute, inp)
        self.log('Done computing voxel based radiomic features!')

    def deletePartialData(self, kernelWidth=5, binWidth=25, absolute=True, inp='t1'):
        self.log('Started deleting partial data!')
        feature_classes = np.array(['firstorder','glcm','glszm','glrlm','ngtdm','gldm'])
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        for n in names:
            for f in feature_classes:
                p = '{}/{}/preprocessed/{}/{}_radiomics_raw_k{}_b{}_{}.npy'.format(self.path,self.space,n,inp,kernelWidth,binstr,f)
                if os.path.exists(p):
                    os.remove(p)
        self.log('Done deleting partial data!')

    def radiomics(self, binWidth=25, absolute=True, inp='t1'):
        features = computeRadiomicsFeatureNames(['firstorder','glcm','glszm','glrlm','ngtdm','gldm','shape'])
        np.save(self.path+'/preprocessed/features',features)
        del features
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        self.log('Started computing radiomic features for {} datapoints on {} core{}!'.format(len(names),self.cores,'s' if self.cores > 1 else ''))
        datapoints = [DataPoint(n,self.path+'/'+self.space,self.debug,self.out,self.visualize) for n in names]
        with multiprocessing.Pool(self.cores) as pool:
            pool.map(wrapperRadiomics, [[d,binWidth,absolute,inp] for d in datapoints])
        self.log('Done computing radiomic features!')
    
    def scaleRadiomics(self, binWidth=25, absolute=True, inp='t1'):
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        self.log('Started computing scale factors for radiomics!')
        features = np.load(self.path+'/preprocessed/features.npy')
        mi = np.repeat(np.array([sys.maxsize],np.float32),len(features))
        ma = np.repeat(np.array([-sys.maxsize],np.float32),len(features))
        for n in names:
            for a in ['t1_mask','roi','targets']:
                arr = np.load(self.path+'/'+self.space+'/preprocessed/{}/{}_radiomics_raw_b{}_{}.npy'.format(n,inp,binstr,a))
                if len(arr.shape) == 1:
                    arr = np.expand_dims(arr, 0)
                mi = np.min(np.concatenate([np.expand_dims(mi,0),np.expand_dims(np.min(arr,0),0)],0),0)
                ma = np.max(np.concatenate([np.expand_dims(ma,0),np.expand_dims(np.max(arr,0),0)],0),0)
        factors = np.concatenate([np.expand_dims(mi,-1),np.expand_dims(ma,-1)],-1)
        np.save(self.path+'/'+self.space+'/preprocessed/{}_features_scale_b{}'.format(inp,binstr), factors)
        del mi; del ma; del factors
        self.log('Done computing scale factors for radiomics!')

    def scaleRadiomicsVoxel(self, kernelWidth=5, binWidth=25, absolute=True, inp='t1'):
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        self.log('Started computing scale factors for voxel based radiomics!')
        features_vox = np.load(self.path+'/preprocessed/features_vox.npy')
        factors_vox = []
        distributions = []
        for i in range(len(features_vox)):
            self.log('Started feature {}!'.format(features_vox[i]))
            con = []
            for j in range(len(names)):
                raw = np.load(self.path+'/'+self.space+'/preprocessed/{}/{}_radiomics_raw_k{}_b{}.npy'.format(names[j],inp,kernelWidth,binstr),mmap_mode='r')
                con.append(raw[:,i])
            con = np.concatenate(con,0)
            f, dis = scaleRadiomics(con)
            if self.visualize:
                showRadiomicsDist(features_vox[i],dis[0:2],dis[2:4],f[2]=='log10')
            factors_vox.append(f)
            distributions.append(dis)
            self.log('Done feature {}!'.format(features_vox[i]))
        np.save(self.path+'/'+self.space+'/preprocessed/{}_features_scale_vox_distributions_k{}_b{}'.format(inp,kernelWidth,binstr), np.array(distributions))
        np.save(self.path+'/'+self.space+'/preprocessed/{}_features_scale_vox_k{}_b{}'.format(inp,kernelWidth,binstr), np.array(factors_vox))
        self.log('Done computing scale factors for voxel based radiomics!')

    def preloadTarget(self):
        path = self.path+'/'+self.space

        self.log('Started preloading data!')
        for i in range(len(self.names)):
            name = self.names[i]
            self.log('Started preloading {}!'.format(name))
            mask = la.load(path+'/preprocessed/{}/mask_basal.pkl'.format(name))
            mask_left = mask[:,:,:,0].flatten()
            mask_right = mask[:,:,:,1].flatten()
            #t1
            t1 = np.load(path+'/preprocessed/{}/t1.npy'.format(name))
            t1_flat_left = np.zeros((np.count_nonzero(mask_left),1),np.float16)
            t1_flat_right = np.zeros((np.count_nonzero(mask_right),1),np.float16)
            slc = t1[:,:,:].flatten()
            t1_flat_left[:,0] = slc[mask_left]
            t1_flat_right[:,0] = slc[mask_right]
            #connectivity
            con = la.load(path+'/preprocessed/{}/connectivity.pkl'.format(name))
            con_flat_left = np.zeros((np.count_nonzero(mask_left),con.shape[-1]),np.float16)
            con_flat_right = np.zeros((np.count_nonzero(mask_right),con.shape[-1]),np.float16)
            for j in range(con.shape[-1]):
                slc = con[:,:,:,j].flatten()
                con_flat_left[:,j] = slc[mask_left]
                con_flat_right[:,j] = slc[mask_right]
            #streamline
            sed = la.load(path+'/preprocessed/{}/streamline.pkl'.format(name))
            sed_flat_left = np.zeros((np.count_nonzero(mask_left),sed.shape[-1]),np.float16)
            sed_flat_right = np.zeros((np.count_nonzero(mask_right),sed.shape[-1]),np.float16)
            for j in range(sed.shape[-1]):
                slc = sed[:,:,:,j].flatten()
                sed_flat_left[:,j] = slc[mask_left]
                sed_flat_right[:,j] = slc[mask_right]
            #basal_seg
            if os.path.exists(path+'/preprocessed/{}/basal_seg.pkl'.format(name)):
                seg = la.load(path+'/preprocessed/{}/basal_seg.pkl'.format(name))
                seg_flat_left = np.zeros((np.count_nonzero(mask_left),seg.shape[-1]),np.float16)
                seg_flat_right = np.zeros((np.count_nonzero(mask_right),seg.shape[-1]),np.float16)
                for j in range(seg.shape[-1]):
                    slc = seg[:,:,:,j].flatten()
                    seg_flat_left[:,j] = slc[mask_left]
                    seg_flat_right[:,j] = slc[mask_right]
            self.log('Saving {}!'.format(name))
            if not os.path.isdir(path+'/preloaded/'+name):
                self.log('Creating output directory at \'{}\'!'.format(path+'/preloaded/'+name))
                os.makedirs(path+'/preloaded/'+name,exist_ok=True)
            np.save(path+'/preloaded/{}/t1_left.npy'.format(name),t1_flat_left)
            np.save(path+'/preloaded/{}/t1_right.npy'.format(name),t1_flat_right)
            np.save(path+'/preloaded/{}/connectivity_left.npy'.format(name),con_flat_left)
            np.save(path+'/preloaded/{}/connectivity_right.npy'.format(name),con_flat_right)
            np.save(path+'/preloaded/{}/streamline_left.npy'.format(name),sed_flat_left)
            np.save(path+'/preloaded/{}/streamline_right.npy'.format(name),sed_flat_right)
            if os.path.exists(path+'/preprocessed/{}/basal_seg.pkl'.format(name)):
                np.save(path+'/preloaded/{}/basal_seg_left.npy'.format(name),seg_flat_left)
                np.save(path+'/preloaded/{}/basal_seg_right.npy'.format(name),seg_flat_right)
            self.log('Done preloading {}!'.format(name))
        self.log('Done preloading data!')

    def preloadRadiomicsVoxel(self, kernelWidth=5, binWidth=25, absolute=True, inp='t1'):
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        path = self.path+'/'+self.space
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')

        self.log('Started preloading data!')
        factors_vox = np.load(path+'/preprocessed/{}_features_scale_vox_k{}_b{}.npy'.format(inp,kernelWidth,binstr))
        for i in range(len(names)):
            name = names[i]
            self.log('Started preloading {}!'.format(name))
            raw = np.load(path+'/preprocessed/{}/{}_radiomics_raw_k{}_b{}.npy'.format(name,inp,kernelWidth,binWidth,binstr))
            mask = la.load(path+'/preprocessed/{}/mask_basal.pkl'.format(name))
            mask_left_cnt = np.count_nonzero(mask[:,:,:,0])
            mask_right_cnt = np.count_nonzero(mask[:,:,:,1])
            res_flat_norm_left = np.zeros((mask_left_cnt,len(factors_vox)),np.float16)
            res_flat_norm_right = np.zeros((mask_right_cnt,len(factors_vox)),np.float16)
            res_flat_scale_left = np.zeros((mask_left_cnt,len(factors_vox)),np.float16)
            res_flat_scale_right = np.zeros((mask_right_cnt,len(factors_vox)),np.float16)
            for j in range(len(factors_vox)):
                norm = raw[:,j]
                scale = raw[:,j]
                if factors_vox[j][2] == 'log10':
                    norm = np.log10(norm+1)
                    fac_norm = np.array(factors_vox[j][3:5],np.float32)
                else:
                    fac_norm = np.array(factors_vox[j][0:2],np.float32)
                fac_scale = np.array(factors_vox[j][0:2],np.float32)
                norm = np.array((norm-fac_norm[0])/(fac_norm[1]-fac_norm[0]),np.float16)
                scale = np.array((scale-fac_scale[0])/(fac_scale[1]-fac_scale[0]),np.float16)
                norm = norm.flatten()
                scale = scale.flatten()
                res_flat_norm_left[:,j] = norm[:mask_left_cnt]
                res_flat_norm_right[:,j] = norm[mask_left_cnt:]
                res_flat_scale_left[:,j] = scale[:mask_left_cnt]
                res_flat_scale_right[:,j] = scale[mask_left_cnt:]
            self.log('Saving {}!'.format(name))
            if not os.path.isdir(path+'/preloaded/'+name):
                self.log('Creating output directory at \'{}\'!'.format(path+'/preloaded/'+name))
                os.makedirs(path+'/preloaded/'+name,exist_ok=True)
            np.save(path+'/preloaded/{}/{}_radiomics_norm_left_k{}_b{}.npy'.format(name,inp,kernelWidth,binstr),res_flat_norm_left)
            np.save(path+'/preloaded/{}/{}_radiomics_norm_right_k{}_b{}.npy'.format(name,inp,kernelWidth,binstr),res_flat_norm_right)
            np.save(path+'/preloaded/{}/{}_radiomics_scale_left_k{}_b{}.npy'.format(name,inp,kernelWidth,binstr),res_flat_scale_left)
            np.save(path+'/preloaded/{}/{}_radiomics_scale_right_k{}_b{}.npy'.format(name,inp,kernelWidth,binstr),res_flat_scale_right)
            self.log('Done preloading {}!'.format(name))
        self.log('Done preloading data!')

    def preloadRadiomics(self, binWidth=25, absolute=True, inp='t1'):
        names = [n for n in self.names if inp not in self.missing.keys() or n not in self.missing[inp]]
        path = self.path+'/'+self.space
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')

        self.log('Started preloading data!')
        factors = np.load(path+'/preprocessed/{}_features_scale_b{}.npy'.format(inp,binstr))
        for i in range(len(names)):
            name = names[i]
            self.log('Started preloading {}!'.format(name))
            tar = np.load(path+'/preprocessed/{}/{}_radiomics_raw_b{}_targets.npy'.format(name,inp,binstr))
            roi = np.load(path+'/preprocessed/{}/{}_radiomics_raw_b{}_roi.npy'.format(name,inp,binstr))
            bra = np.load(path+'/preprocessed/{}/{}_radiomics_raw_b{}_t1_mask.npy'.format(name,inp,binstr))
            mi = np.expand_dims(factors[:,0],0)
            ma = np.expand_dims(factors[:,1],0)
            bra = np.expand_dims(bra,0)
            tar = np.array((tar-np.repeat(mi,len(tar),0))/np.repeat((ma-mi),len(tar),0),np.float16)
            roi = np.array((roi-np.repeat(mi,len(roi),0))/np.repeat((ma-mi),len(roi),0),np.float16)
            bra = np.array((bra-np.repeat(mi,len(bra),0))/np.repeat((ma-mi),len(bra),0),np.float16)
            self.log('Saving {}!'.format(name))
            if not os.path.isdir(path+'/preloaded/'+name):
                self.log('Creating output directory at \'{}\'!'.format(path+'/preloaded/'+name))
                os.makedirs(path+'/preloaded/'+name,exist_ok=True)
            np.save(path+'/preloaded/{}/{}_radiomics_scale_b{}_targets.npy'.format(name,inp,binstr),tar)
            np.save(path+'/preloaded/{}/{}_radiomics_scale_b{}_roi.npy'.format(name,inp,binstr),roi)
            np.save(path+'/preloaded/{}/{}_radiomics_scale_b{}_t1_mask.npy'.format(name,inp,binstr),bra)
            self.log('Done preloading {}!'.format(name))
        self.log('Done preloading data!')
