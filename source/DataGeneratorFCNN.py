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
        left          = True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = False,      #include right hemisphere data
        threshold     = 0.6,        #if float value provided, it thresholds the connectivty map
        binarize      = True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
        not_connected = True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
        background    = True,
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
        shape         = None,
        debug         = False,
    ):
        self.debug = debug
        self.path = path
        self.seed = seed
        self.split = split
        self.test_split= test_split
        self.control = control
        self.huntington = huntington
        self.names = self.getSplit()
        self.left = left
        self.right = right
        self.threshold = threshold
        self.binarize = binarize
        self.not_connected = not_connected
        self.background = background
        self.features_vox = features_vox
        self.feature_mask_vox = self.getFeatureMask(features_vox,np.load(path+'/preprocessed/features_vox.npy'))
        self.radiomics_vox = radiomics_vox
        if shape is None:
            self.shape = tuple(np.max(np.load(self.path+'/preprocessed/shapes.npy'),0))
        else:
            self.shape = shape

    def getData(self):
        return [self.getDatapoints(n) for n in self.names]

    def getDatapoints(self, names):
        x = None
        y = None
        bg = None
        for i in range(len(names)):
            xi = self.getVox(names[i])
            yi = self.getCon(names[i])
            bgi = np.load('{}/preprocessed/{}/t1_mask.npy'.format(self.path,names[i]))
            if x is None:
                x = np.zeros((len(names),)+self.shape+(xi.shape[-1],),np.float16)
                y = np.zeros((len(names),)+self.shape+(yi.shape[-1],),np.float16)
                bg = np.zeros((len(names),)+self.shape,np.bool_)
            center = (np.array(self.shape)-np.array(xi.shape[0:3]))//2
            x[i,center[0]:center[0]+xi.shape[0],center[1]:center[1]+xi.shape[1],center[2]:center[2]+xi.shape[2],:] = xi
            y[i,center[0]:center[0]+yi.shape[0],center[1]:center[1]+yi.shape[1],center[2]:center[2]+yi.shape[2],:] = yi
            bg[i,center[0]:center[0]+bgi.shape[0],center[1]:center[1]+bgi.shape[1],center[2]:center[2]+bgi.shape[2]] = bgi
        return [x,y,bg,names]

    def getVox(self, name):
        return np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_{}.npy'.format(self.path,name,rad))[:,:,:,self.feature_mask_vox] for rad in self.radiomics_vox],-1)

    def getCon(self, name):
        raw = la.load(self.path+'/preprocessed/'+name+'/connectivity.pkl')
        raw = self.getHemispheres(raw)
        if self.threshold is not None:
            raw = np.where(raw <= self.threshold, 0, raw)
            if self.binarize:
                raw = convertToMask(raw)
        if self.not_connected and self.threshold is not None and self.threshold >= 0.5:
            mask = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
            mask = self.getHemispheres(mask)
            mask = np.transpose(mask,[3,0,1,2])
            mask = np.logical_or.reduce(mask)
            if raw.dtype == np.bool_:
                nc = np.transpose(raw,[3,0,1,2])
                nc = np.logical_or.reduce(nc)
                nc = np.logical_xor(nc,mask)
            else:
                nc = (np.sum(raw, axis=-1)*-1)+1
                nc = np.where(mask, nc, 0)
            nc = np.expand_dims(nc, -1)
            raw = np.concatenate([raw,nc],-1)
            if self.background:
                bg = np.logical_not(mask)
                bg = np.expand_dims(bg, -1)
                raw = np.concatenate([raw,bg],-1)
        return np.array(raw,np.float16)
    
    def getHemispheres(self, data):
        if self.left and self.right:
            return data
        half = data.shape[-1]//2
        if self.left and not self.right:
            return data[:,:,:,:half]
        if not self.left and self.right:
            return data[:,:,:,half:]
        if data.dtype == np.bool_:
            return np.logical_or(data[:,:,:,:half],data[:,:,:,half:])
        return data[:,:,:,:half]+data[:,:,:,half:]
    
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
        huns_train = huns[:int(len(huns)*s0)]
        huns_test  = huns[int(len(huns)*s0):int(len(huns)*s1)]
        huns_val   = huns[int(len(huns)*s1):]
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
        if self.debug:
            return [tr[0:1], va[0:1], te[0:1]]
        return [tr, va, te]

    @staticmethod
    def getFeatureMask(features, raw_features):
        if features is None or len(features) == 0:
            features = raw_features
        feature_mask = []
        for i in range(len(raw_features)):
            feature_mask.append(raw_features[i] in features)
        return np.array(feature_mask, np.bool_)