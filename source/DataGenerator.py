import numpy as np
import LayeredArray as la
from util import convertToMask

class DataGenerator():
    def __init__(self,
        path          = 'data',     #path of the data
        seed          = 42,         #seed for the split
        split         = 0.8,        #train/all ratio
        test_split    = 0.5,        #test/(test+validation) ratio
        control       = True,       #include control data points
        huntington    = False,      #include huntington data points
        spatial       = False,      #keep spaital format of flatten voxels in the brain region
        left          = True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = False,      #include right hemisphere data
        threshold     = 0.5,        #if float value provided, it thresholds the connectivty map
        binarize      = True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
        not_connected = True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
        single        = None,       #if int index value is provided, it only returns a specified connectivity map
        target        = False,
        roi           = False,
        brain         = False,
        features      = [],         #used radiomics features (emptylist means all)
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics     = ['b25'],    #used radiomics features bin settings
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
        balance_data  = True,
    ):
        self.path = path
        self.seed = seed
        self.split = split
        self.test_split= test_split
        self.control = control
        self.huntington = huntington
        self.names = self.getSplit()
        self.spatial = spatial
        self.left = left
        self.right = right
        self.threshold = threshold
        self.binarize = binarize
        self.not_connected = not_connected
        self.single = None
        if single is not None:
            if isinstance(single,list):
                self.single = single
            else:
                self.single = [single]
        self.target = target
        self.roi = roi
        self.brain = brain
        self.feature_mask = self.getFeatureMask(features,np.load(path+'/preprocessed/features.npy'))
        self.feature_mask_vox = self.getFeatureMask(features_vox,np.load(path+'/preprocessed/features_vox.npy'))
        self.radiomics = radiomics
        self.radiomics_vox = radiomics_vox
        self.balance_data = balance_data
        labels = np.load(self.path+'/preprocessed/labels.npy')
        if self.single is not None:
            l = len(self.single)
        else:
            l = len(labels)*2
            if (not self.left) or (not self.right):
                l = l//2
            if self.not_connected and self.threshold is not None and self.threshold >= 0.5:
                l = l+1
        if self.spatial:
            shapes = np.load(self.path+'/preprocessed/shapes.npy')
            self.shape = tuple(np.max(shapes,0))
            self.x_shape = self.shape+(np.count_nonzero(self.feature_mask_vox)*len(self.radiomics_vox),)
            self.y_shape = self.shape+(l,)
        else:
            self.x_shape = (np.count_nonzero(self.feature_mask_vox)*len(self.radiomics_vox),)
            self.y_shape = (l,)

    def getData(self):
        return [self.getDatapoints(n) for n in self.names]
    
    def getDatapoints(self, names):
        data = [self.getDatapoint(n) for n in names]
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        if self.spatial:
            x = np.array(x)
            y = np.array(y)
        else:
            x = np.concatenate(x,0)
            y = np.concatenate(y,0)
        return [x, y]

    def getDatapoint(self, name):
        x = self.getVox(name)
        y = self.getCon(name)
        if self.spatial:
            x = [x]
            if self.target:
                x.append(self.getOth(name,'targets'))
            if self.roi:
                x.append(self.getOth(name,'roi'))
            if self.brain:
                x.append(self.getOth(name,'t1_mask'))
            x = np.concatenate(x,-1)
            center = (np.array(self.shape)-np.array(x.shape[0:3]))//2
            tmp = np.zeros(self.shape+(x.shape[-1],),np.float16)
            tmp[center[0]:center[0]+x.shape[0],center[1]:center[1]+x.shape[1],center[2]:center[2]+x.shape[2],:] = x
            x = tmp
            tmp = np.zeros(self.shape+(y.shape[-1],),np.float16)
            tmp[center[0]:center[0]+x.shape[0],center[1]:center[1]+x.shape[1],center[2]:center[2]+x.shape[2],:] = y
            y = tmp
        else:
            if self.balance_data:
                dat = y
                if self.single is None and self.not_connected and self.threshold is not None and self.threshold >= 0.5:
                    dat = dat[:,0:-1]
                dat = np.max(dat,1)
                positive_idxs = np.argwhere(dat >= 0.5).T[0]
                positive_cnt = len(positive_idxs)
                negative_cnt = len(y)-positive_cnt
                remainder = negative_cnt % positive_cnt
                positive_y = y[positive_idxs,:]
                positive_x = x[positive_idxs,:]
                y = [y,np.repeat(positive_y,negative_cnt//positive_cnt,0)]
                x = [x,np.repeat(positive_x,negative_cnt//positive_cnt,0)]
                if remainder > 0: y += [positive_y[0:remainder,:]]
                if remainder > 0: x += [positive_x[0:remainder,:]]
                y = np.concatenate(y,0)
                x = np.concatenate(x,0)
            x = [x]
            if self.target:
                x.append(np.repeat(np.expand_dims(self.getOth(name,'targets').flatten(),0),len(x),0))
            if self.roi:
                x.append(np.repeat(np.expand_dims(self.getOth(name,'roi').flatten(),0),len(x),0))
            if self.brain:
                x.append(np.repeat(np.expand_dims(self.getOth(name,'t1_mask').flatten(),0),len(x),0))
            x = np.concatenate(x,-1)
        return [x, y]

    def getVox(self, name):
        if self.spatial:
            return np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_{}.npy'.format(self.path,name,rad))[:,:,:,self.feature_mask_vox] for rad in self.radiomics_vox],-1)
        else:
            raw = []
            if self.left or ((not self.left) and (not self.right)):
                raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_left_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
            if self.right or ((not self.left) and (not self.right)):
                raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_right_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
            return np.concatenate(raw,0)

    def getCon(self, name):
        if self.spatial:
            raw = la.load(self.path+'/preprocessed/'+name+'/connectivity.pkl')
            raw = self.getHemispheres(raw)
            if self.single is not None:
                raw = raw[:,:,:,self.single]
        else:
            raw = []
            if self.left or ((not self.left) and (not self.right)):
                raw.append(np.load('{}/preloaded/{}/connectivity_left.npy'.format(self.path,name)))
            if self.right or ((not self.left) and (not self.right)):
                raw.append(np.load('{}/preloaded/{}/connectivity_right.npy'.format(self.path,name)))
            raw = np.concatenate(raw,0)
            if self.single is not None:
                raw = raw[:,self.single]
        if self.threshold is not None:
            raw = np.where(raw <= self.threshold, 0, raw)
            if self.binarize:
                raw = convertToMask(raw)
        if self.single is None and self.not_connected and self.threshold is not None and self.threshold >= 0.5:
            if self.spatial:
                mask = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
                mask = self.getHemispheres(mask)
            if raw.dtype == np.bool_:
                if self.spatial:
                    nc = np.transpose(raw,[3,0,1,2])
                    nc = np.logical_or.reduce(nc)
                    nc = np.logical_xor(nc,mask)
                else:
                    nc = np.transpose(raw,[1,0])
                    nc = np.logical_or.reduce(nc)
                    nc = np.logical_not(nc)
            else:
                nc = (np.sum(raw, axis=-1)*-1)+1
                if self.spatial:
                    nc = np.where(mask, nc, 0)
            nc = np.expand_dims(nc, -1)
            raw = np.concatenate([raw,nc],-1)
        return np.array(raw,np.float16)
    
    def getOth(self, name, file):
        if self.spatial:
            if file in ['targets','roi']:
                raw = la.load(self.path+'/preprocessed/'+name+'/'+file+'.pkl')
                raw = self.getHemispheres(raw)
                if self.single is not None:
                    raw = raw[:,:,:,self.single]
                return np.array(raw,np.float16)
            else:
                raw = np.load(self.path+'/preprocessed/'+name+'/'+file+'.npy')
                if len(raw.shape) == 3:
                    raw = np.expand_dims(raw,-1)
                return np.array(raw,np.float16)
        else:
            raw = np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_scale_{}_{}.npy'.format(self.path,name,rad,file))[:,self.feature_mask] for rad in self.radiomics],-1)
            raw = self.getHemispheres(raw)
            if self.single is not None:
                raw = raw[self.single,:]
            return raw
    
    def getHemispheres(self, data):
        if self.left and self.right:
            return data
        if len(data.shape) == 4:
            half = data.shape[-1]//2
            if self.left and not self.right:
                return data[:,:,:,:half]
            if not self.left and self.right:
                return data[:,:,:,half:]
            if data.dtype == np.bool_:
                return np.logical_or(data[:,:,:,:half],data[:,:,:,half:])
            return data[:,:,:,:half]+data[:,:,:,half:]
        if len(data.shape) == 2:
            half = data.shape[0]//2
            if self.left and not self.right:
                return data[:half,:]
            if not self.left and self.right:
                return data[half:,:]
            return data[:half,:]+data[half:,:]
    
    def getSplit(self):
        if not self.control and not self.huntington:
            raise Exception('Must include control and/or huntington data points!')
        names = np.load(self.path+'/preprocessed/names.npy')
        cons = [n for n in names if n[0] == 'C']
        huns = [n for n in names if n[0] == 'H']
        ran = np.random.default_rng(self.seed)
        ran.shuffle(cons)
        ran.shuffle(huns)
        s0 = self.split
        s1 = self.split+(1-self.split)*self.test_split
        cons_train = cons[:int(len(cons)*s0)]
        cons_test  = cons[int(len(cons)*s0):int(len(cons)*s1)]
        cons_val   = cons[int(len(cons)*s1):]
        huns_train = cons[:int(len(huns)*s0)]
        huns_test  = cons[int(len(huns)*s0):int(len(huns)*s1)]
        huns_val   = cons[int(len(huns)*s1):]
        tr = []
        te = []
        va = []
        if self.control:
            tr = tr+cons_train
            te = te+cons_test
            va = va+cons_val
        if self.huntington:
            tr = tr+huns_train
            te = te+huns_test
            va = va+huns_val
        ran.shuffle(tr)
        ran.shuffle(te)
        ran.shuffle(va)
        return [tr, te, va]

    @staticmethod
    def getFeatureMask(features, raw_features):
        if features is None or len(features) == 0:
            features = raw_features
        feature_mask = []
        for i in range(len(raw_features)):
            feature_mask.append(raw_features[i] in features)
        return np.array(feature_mask, np.bool_)