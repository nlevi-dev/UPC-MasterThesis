import numpy as np
import LayeredArray as la
from util import convertToMask
from sklearn.decomposition import PCA

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
        threshold     = 0.6,        #if float value provided, it thresholds the connectivty map, if 0 int proveded it re-one-hot encodes it
        binarize      = True,       #binarizes the connectivity map
        not_connected = True,       #appends an extra encoding for the 'not connected' label
        single        = None,       #returns only a single label layer
        target        = False,      #includes target region(s) [all if not single] in the x values
        roi           = False,      #includes roi region(s) in the x values
        brain         = False,      #includes entire brain in the x values
        features      = [],         #used radiomics features (emptylist means all)
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics     = ['b25'],    #used radiomics features bin settings
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
        output        = 'connectivity.pkl',
        balance_data  = True,       #balances data
        debug         = False,      #if true, it only return 1-1-1 datapoints for train-val-test
        targets_all   = False,      #includes all target regions regardless if single or not
        collapse_max  = False,      #collapses the last dimesnion with maximum function (used for regression)
        extras        = None,       #includes extra data for each datapoint (format {'datapoint_name':[data]})
        pca           = None,       #if provided a float value it keeps that fraction of the explained variance
        pca_parts     = None,       #only applies PCA to parts of the input space, possible values: [vox,target,roi,brain]
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
        self.single = single
        self.target = target
        self.roi = roi
        self.brain = brain
        self.features = features
        self.features_vox = features_vox
        self.features_raw = np.load(path+'/preprocessed/features.npy')
        self.features_vox_raw = np.load(path+'/preprocessed/features_vox.npy')
        self.feature_mask = self.getFeatureMask(self.features,self.features_raw)
        self.feature_mask_vox = self.getFeatureMask(self.features_vox,self.features_vox_raw)
        self.radiomics = radiomics
        self.radiomics_vox = radiomics_vox
        self.output = output
        self.balance_data = balance_data
        self.extras = extras
        self.targets_all = targets_all
        self.collapse_max = collapse_max
        self.pca = pca
        self.pca_obj = None
        self.pca_comps = None
        self.pca_parts = pca_parts
        self.pca_range = None

    def getData(self):
        if self.pca is not None and self.pca_obj is None:
            train = self.getDatapoints(self.names[0])
            self.pca_obj = PCA().fit(train[0] if self.pca_range is None else train[0][:,self.pca_range])
            cnt = 0
            self.pca_comps = 0
            while cnt < self.pca:
                cnt += self.pca_obj.explained_variance_ratio_[self.pca_comps]
                self.pca_comps += 1
        return [self.getDatapoints(n) for n in self.names]
    
    def getReconstructor(self, name, xy_only=False):
        x, y = self.getDatapoint(name, balance_override=True)
        if xy_only:
            return [x, y]
        mask = la.load(self.path+'/preprocessed/{}/roi.pkl'.format(name))
        mask_left = mask[:,:,:,0].flatten()
        mask_right = mask[:,:,:,1].flatten()
        mask_left = np.argwhere(mask_left).T[0]
        mask_right = np.argwhere(mask_right).T[0]
        idxs = []
        if self.left or ((not self.left) and (not self.right)):
            idxs.append(mask_left)
        if self.right or ((not self.left) and (not self.right)):
            idxs.append(mask_right)
        idxs = np.concatenate(idxs,0)
        bg = np.load(self.path+'/preprocessed/{}/t1_mask.npy'.format(name))
        return [x, y, idxs, bg, name]

    def getDatapoints(self, names):
        data = [self.getDatapoint(n) for n in names]
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        x = np.concatenate(x,0)
        y = np.concatenate(y,0)
        np.random.default_rng(self.seed).shuffle(x)
        np.random.default_rng(self.seed).shuffle(y)
        return [x, y]

    def getDatapoint(self, name, balance_override=False):
        x = self.getVox(name)
        if self.pca_parts == 'vox' and self.pca_range is None:
            self.pca_range = range(0,x.shape[-1])
        if self.pca_parts == 'vox' and self.pca_obj is not None and self.pca_range is not None:
            app = self.pca_obj.transform(app)[:,0:self.pca_comps]
        y = self.getCon(name)
        if self.balance_data and not balance_override:
            dat = y
            if self.not_connected and self.threshold is not None and self.threshold >= 0.5:
                dat = dat[:,0:-1]
            positive_idxs = np.argwhere(np.max(dat,1) >= 0.5).T[0]
            negative_cnt = len(y)-len(positive_idxs)
            for i in range(dat.shape[-1]):
                positive_idxs = np.argwhere(dat[:,i] >= 0.5).T[0]
                positive_cnt = len(positive_idxs)
                if positive_cnt == 0:
                    #print('ZERO POSITIVE LABELS at {} {}'.format(name,i))
                    continue
                remainder = negative_cnt % positive_cnt
                positive_y = np.take(y,positive_idxs,0)
                positive_x = np.take(x,positive_idxs,0)
                y = [y,np.repeat(positive_y,negative_cnt//positive_cnt,0)]
                x = [x,np.repeat(positive_x,negative_cnt//positive_cnt,0)]
                if remainder > 0: y += [np.take(positive_y,range(0,remainder),0)]
                if remainder > 0: x += [np.take(positive_x,range(0,remainder),0)]
                y = np.concatenate(y,0)
                x = np.concatenate(x,0)
        x1 = [x]
        if self.target and len(self.radiomics) > 0:
            app = np.repeat(np.expand_dims(self.getOth(name,'targets').flatten(),0),len(x),0)
            if self.pca_parts == 'target' and self.pca_range is None:
                self.pca_range = range(np.sum([e.shape[-1] for e in x1]),np.sum([e.shape[-1] for e in x1])+app.shape[-1])
            if self.pca_parts == 'target' and self.pca_obj is not None and self.pca_range is not None:
                app = self.pca_obj.transform(app)[:,0:self.pca_comps]
            x1.append(app)
        if self.roi and len(self.radiomics) > 0:
            app = np.repeat(np.expand_dims(self.getOth(name,'roi').flatten(),0),len(x),0)
            if self.pca_parts == 'roi' and self.pca_range is None:
                self.pca_range = range(np.sum([e.shape[-1] for e in x1]),np.sum([e.shape[-1] for e in x1])+app.shape[-1])
            if self.pca_parts == 'roi' and self.pca_obj is not None and self.pca_range is not None:
                app = self.pca_obj.transform(app)[:,0:self.pca_comps]
            x1.append(app)
        if self.brain and len(self.radiomics) > 0:
            app = np.repeat(np.expand_dims(self.getOth(name,'t1_mask').flatten(),0),len(x),0)
            if self.pca_parts == 'brain' and self.pca_range is None:
                self.pca_range = range(np.sum([e.shape[-1] for e in x1]),np.sum([e.shape[-1] for e in x1])+app.shape[-1])
            if self.pca_parts == 'brain' and self.pca_obj is not None and self.pca_range is not None:
                app = self.pca_obj.transform(app)[:,0:self.pca_comps]
            x1.append(app)
        x = np.concatenate(x1,-1)
        if self.pca_obj is not None and self.pca_range is None:
            x = self.pca_obj.transform(x)[:,0:self.pca_comps]
        return [x, y]

    def getVox(self, name):
        if len(self.radiomics_vox) == 0 and self.extras is not None:
            return self.extras[name]
        raw = []
        if self.left or ((not self.left) and (not self.right)):
            raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_left_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
        if self.right or ((not self.left) and (not self.right)):
            raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_right_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
        raw = np.concatenate(raw,0)
        if self.extras is not None:
            raw = np.concatenate([raw,self.extras[name]],-1)
        return raw

    def getCon(self, name):
        raw = []
        if self.left or ((not self.left) and (not self.right)):
            raw.append(np.load('{}/preloaded/{}/connectivity_left.npy'.format(self.path,name)))
        if self.right or ((not self.left) and (not self.right)):
            raw.append(np.load('{}/preloaded/{}/connectivity_right.npy'.format(self.path,name)))
        raw = np.concatenate(raw,0)
        raw = self.getHemispheres(raw, -1)
        if self.threshold is not None:
            if self.threshold == 0:
                bin = np.zeros(raw.shape,raw.dtype)
                arged = np.argmax(raw,-1)
                for i in range(bin.shape[-1]):
                    bin[:,i] = np.where(arged == i, 1, 0)
                raw = bin
            else:
                raw = np.where(raw <= self.threshold, 0, raw)
            if self.binarize:
                raw = convertToMask(raw)
        if self.single is not None:
            raw = raw[:,self.single:self.single+1]
        if self.not_connected and self.threshold is not None and self.threshold >= 0.5:
            if raw.dtype == np.bool_:
                nc = np.transpose(raw,[1,0])
                nc = np.logical_or.reduce(nc)
                nc = np.logical_not(nc)
            else:
                nc = (np.sum(raw, axis=-1)*-1)+1
            nc = np.expand_dims(nc, -1)
            raw = np.concatenate([raw,nc],-1)
        if self.collapse_max:
            raw = np.expand_dims(np.max(raw,-1),-1)
        return np.array(raw,np.float16)

    def getOth(self, name, file):
        raw = []
        for i in range(len(self.radiomics)):
            if i > 0 and len(self.features) == 0:
                mask = self.getFeatureMask([f for f in self.features_raw if 'shape' not in f],self.features_raw)
            else:
                mask = self.feature_mask
            raw.append(np.load('{}/preloaded/{}/t1_radiomics_scale_{}_{}.npy'.format(self.path,name,self.radiomics[i],file))[:,mask])
        raw = np.concatenate(raw,-1)
        raw = self.getHemispheres(raw, 0)
        if self.single is not None and file == 'targets' and not self.targets_all:
            raw = raw[self.single:self.single+1,:]
        return raw

    def getHemispheres(self, data, idx=-1):
        if (self.left and self.right) or data.shape[idx] == 1:
            return data
        half = data.shape[idx]//2
        if self.left and not self.right:
            return data[:,:half] if idx == -1 else data[:half,:]
        if not self.left and self.right:
            return data[:,half:] if idx == -1 else data[half:,:]
        return data[:,:half]+data[:,half:] if idx == -1 else data

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
    
    def clonePCA(self, from_obj):
        self.pca = from_obj.pca
        self.pca_obj = from_obj.pca_obj
        self.pca_comps = from_obj.pca_comps
        self.pca_parts = from_obj.pca_parts
        self.pca_range = from_obj.pca_range

def reconstruct(y, idxs, bg):
    return np.concatenate([place(y[:,i],idxs,bg.shape) for i in range(y.shape[-1])],-1)

def place(data, idxs, shape):
    ret = np.zeros(shape,np.float16)
    ret = ret.flatten()
    ret[idxs] = data
    ret = ret.reshape(shape)
    ret = np.expand_dims(ret,-1)
    return ret

