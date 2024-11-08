import numpy as np
import LayeredArray as la

class DataGenerator():
    def __init__(self,
        path          = 'data',     #path of the data
        seed          = 42,         #seed for the split
        split         = 0.8,        #train/all ratio
        test_split    = 0,          #test/(test+validation) ratio
        control       = False,      #include control data points
        huntington    = True,       #include huntington data points
        left          = True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = False,      #include right hemisphere data
        single        = 0,          #
        mask          = True,       #
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
        shape         = None,
        debug         = False,
    ):
        self.debug = debug
        self.path = path
        self.seed = seed
        self.split = split
        self.test_split = test_split
        self.control = control
        self.huntington = huntington
        self.names = self.getSplit()
        self.left = left
        self.right = right
        self.single = single
        self.mask = mask
        self.features_vox = features_vox
        self.feature_mask_vox = self.getFeatureMask(features_vox,np.load(path+'/preprocessed/features_vox.npy'))
        self.radiomics_vox = radiomics_vox
        if shape is None:
            self.shape = tuple(np.max(np.load(self.path+'/preprocessed/shapes.npy'),0))
        else:
            self.shape = shape

    def getData(self):
        return [self.getDatapoints(n) for n in self.names]

    def getDatapoints(self, names, include_background=False):
        x = None
        y = None
        m = None
        bg = None
        for i in range(len(names)):
            xi = self.getVox(names[i])
            yi = self.getCon(names[i])
            mi = self.getMas(names[i])
            if x is None:
                x = np.zeros((len(names),)+self.shape+(xi.shape[-1],),np.float16)
                y = np.zeros((len(names),)+self.shape+(yi.shape[-1],),np.float16)
                if self.mask:
                    m = np.zeros((len(names),)+self.shape+(mi.shape[-1],),np.float16)
                if include_background:
                    bg = np.zeros((len(names),)+self.shape,np.float16)
            center = (np.array(self.shape)-np.array(xi.shape[0:3]))//2
            x[i,center[0]:center[0]+xi.shape[0],center[1]:center[1]+xi.shape[1],center[2]:center[2]+xi.shape[2],:] = xi
            y[i,center[0]:center[0]+yi.shape[0],center[1]:center[1]+yi.shape[1],center[2]:center[2]+yi.shape[2],:] = yi
            if self.mask:
                m[i,center[0]:center[0]+mi.shape[0],center[1]:center[1]+mi.shape[1],center[2]:center[2]+mi.shape[2],:] = mi
            if include_background:
                bi = np.load(self.path+'/preprocessed/{}/t1_mask.npy'.format(names[i]))
                bg[i,center[0]:center[0]+bi.shape[0],center[1]:center[1]+bi.shape[1],center[2]:center[2]+bi.shape[2]] = bi
        return [x,y,names,m,bg]

    def getVox(self, name):
        return np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_{}.npy'.format(self.path,name,rad))[:,:,:,self.feature_mask_vox] for rad in self.radiomics_vox],-1)

    def getCon(self, name):
        raw = la.load(self.path+'/preprocessed/'+name+'/streamline.pkl')
        raw = self.getHemispheres(raw)
        if self.single is not None:
            raw = raw[:,:,:,self.single]
            raw = np.expand_dims(raw,-1)
        raw /= 1000
        return raw
    
    def getMas(self, name):
        raw = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
        raw = self.getHemispheres(raw)
        return raw
    
    def getHemispheres(self, data):
        if self.left and self.right:
            if data.dtype == np.bool_:
                return np.logical_or(data[:,:,:,:half],data[:,:,:,half:])
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
        cs0 = int(len(cons)*s0)
        cs1 = int(len(cons)*s1)
        if cs1 > len(cons)-1:
            cs1 = len(cons)-1
        if cs0 >= cs1:
            cs0 = cs1-1
        hs0 = int(len(huns)*s0)
        hs1 = int(len(huns)*s1)
        if hs1 > len(huns)-1:
            hs1 = len(huns)-1
        if hs0 >= hs1:
            hs0 = hs1-1
        cons_train = cons[:cs0]
        cons_test  = cons[cs0:cs1]
        cons_val   = cons[cs1:]
        huns_train = huns[:hs0]
        huns_test  = huns[hs0:hs1]
        huns_val   = huns[hs1:]
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