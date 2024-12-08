import numpy as np
import LayeredArray as la
from util import convertToMask, pickleLoad
from sklearn.decomposition import PCA

class DataGenerator():
    def __init__(self,
        path          = 'data',         #path of the data
        seed          = 42,             #seed for the split
        split         = 0.8,            #train/all ratio
        test_split    = 0.5,            #test/(test+validation) ratio
        control       = True,           #include control data points
        huntington    = False,          #include huntington data points
        left          = True,           #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = False,          #include right hemisphere data
        threshold     = 0.6,            #if float value provided, it thresholds the connectivty map, if 0 int proveded it re-one-hot encodes it
        binarize      = True,           #binarizes the connectivity map
        not_connected = True,           #appends an extra encoding for the 'not connected' label
        single        = None,           #returns only a single label layer
        features      = [],             #used radiomics features (emptylist means all)
        features_vox  = [],             #used voxel based radiomics features (emptylist means all)
        radiomics     = [               #non-voxel based input space, image, bin and file selection
            {'sp':'native','im':'t1','fe':['b10','b25','b50','b75'],'fi':['targets','roi','t1_mask']},
        ],
        space         = 'native',       #voxel based input and output space selection (native/normalized)
        radiomics_vox = [               #voxel based input image selection and bin settings
            {'im':'t1','fe':['k5_b25']},
        ],
        rad_vox_norm  = 'norm',         #norm/scale
        inps          = [],             #t1/t1t2/diffusion/diffusion_fa/diffusion_md/diffusion_rd
        features_clin = None,           #include clinical data (empty array means all)
        outp          = 'connectivity', #output type selection (connectivity/streamline/basal_seg)
        balance_data  = True,           #balances data
        debug         = False,          #if true, it only return 1-1-1 datapoints for train-val-test
        targets_all   = False,          #includes all target regions regardless if single or not
        collapse_max  = False,          #collapses the last dimesnion with maximum function (used for regression)
        collapse_bin  = False,          #binarizes the collapsed output layer
        extras        = None,           #includes extra data for each datapoint (format {'datapoint_name':[data]})
        exclude       = [],             #excludes names from the missing object
        pca           = None,           #if provided a float value it keeps that fraction of the explained variance
        pca_parts     = None,           #only applies PCA to parts of the input space, possible values: [vox,target,roi,brain]
    ):
        if outp == 'basal_seg' and huntington:
            raise Exception('Error: basal_seg not available for huntington datapoints!')
        if features_clin is not None and control:
            raise Exception('Error: clinical data not available for control datapoints!')
        if pca is not None or pca_parts is not None:
            raise Exception('PCA not implemented error!')
        self.debug = debug
        self.path = path
        self.seed = seed
        self.split = split
        self.test_split= test_split
        self.control = control
        self.huntington = huntington
        self.radiomics = radiomics
        self.radiomics_vox = radiomics_vox
        self.inps = inps
        self.outp = outp
        self.space = space
        self.features_clin_raw = np.load(path+'/preprocessed/features_clinical.npy')
        self.features_clin = [] if features_clin is None else (features_clin if len(features_clin) > 0 else self.features_clin_raw)
        self.exclude = exclude
        self.names = self.getSplit()
        self.left = left
        self.right = right
        self.threshold = threshold
        self.binarize = binarize
        self.not_connected = not_connected
        self.single = single
        self.features_raw = np.load(path+'/preprocessed/features.npy')
        self.features_vox_raw = np.load(path+'/preprocessed/features_vox.npy')
        self.features = features if len(features) > 0 else self.features_raw
        self.features_vox = features_vox if len(features_vox) > 0 else self.features_vox_raw
        self.feature_mask = self.getFeatureMask(self.features,self.features_raw)
        self.feature_mask_shapeless = self.getFeatureMask([f for f in self.features if 'shape' not in f],self.features_raw)
        self.feature_mask_vox = self.getFeatureMask(self.features_vox,self.features_vox_raw)
        self.feature_mask_clin = self.getFeatureMask(self.features_clin,self.features_clin_raw)
        self.rad_vox_norm = rad_vox_norm
        self.balance_data = balance_data
        self.extras = extras
        self.targets_all = targets_all
        self.collapse_max = collapse_max
        self.collapse_bin = collapse_bin
        self.pca = pca
        self.pca_obj = None
        self.pca_comps = None
        self.pca_parts = pca_parts
        self.pca_range = None

    def getData(self, cnt=3):
        if self.pca is not None and self.pca_obj is None:
            train = self.getDatapoints(self.names[0])
            self.pca_obj = PCA().fit(train[0] if self.pca_range is None else train[0][:,self.pca_range])
            cnt = 0
            self.pca_comps = 0
            while cnt < self.pca:
                cnt += self.pca_obj.explained_variance_ratio_[self.pca_comps]
                self.pca_comps += 1
        return [self.getDatapoints(n) for n in self.names[:cnt]]
    
    def getReconstructor(self, name, xy_only=False):
        x, y = self.getDatapoint(name, balance_override=True)
        if xy_only:
            return [x, y]
        mask = la.load(self.path+'/'+self.space+'/preprocessed/{}/mask_basal.pkl'.format(name))
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
        bg = np.load(self.path+'/'+self.space+'/preprocessed/{}/mask_brain.npy'.format(name))
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
            if (self.binarize and self.threshold is not None and (self.threshold >= 0.5 or self.threshold == 0)) or self.outp == 'basal_seg':
                negative_cnt = np.max(np.count_nonzero(dat,0))
            else:
                if self.not_connected and self.threshold is not None and self.threshold >= 0.5:
                    dat = dat[:,0:-1]
                thr = 0.5
                if self.threshold is not None and self.threshold >= 0.5:
                    thr = self.threshold
                negative_cnt = len(y)-np.count_nonzero(np.max(dat,1) >= thr)
            for i in range(dat.shape[-1]):
                positive_idxs = np.argwhere(dat[:,i] >= 0.5).T[0]
                positive_cnt = len(positive_idxs)
                if positive_cnt == 0:
                    #print('ZERO POSITIVE LABELS at {} {}'.format(name,i))
                    continue
                remainder = negative_cnt % positive_cnt
                div = negative_cnt // positive_cnt
                if (remainder == 0 and div == 0):
                    continue
                positive_y = np.take(y,positive_idxs,0)
                positive_x = np.take(x,positive_idxs,0)
                y = [y,np.repeat(positive_y,div,0)]
                x = [x,np.repeat(positive_x,div,0)]
                if remainder > 0: y += [np.take(positive_y,range(0,remainder),0)]
                if remainder > 0: x += [np.take(positive_x,range(0,remainder),0)]
                y = np.concatenate(y,0)
                x = np.concatenate(x,0)
        x1 = [x]
        if len(self.radiomics) > 0 or len(self.features_clin) > 0:
            app = np.repeat(np.expand_dims(self.getOth(name),0),len(x),0)
            x1.append(app)
        x = np.concatenate(x1,-1)
        if self.pca_obj is not None and self.pca_range is None:
            x = self.pca_obj.transform(x)[:,0:self.pca_comps]
        return [x, y]

    def getVox(self, name):
        if len(self.radiomics_vox) == 0 and self.extras is not None:
            return self.extras[name]
        raw = []
        #[{'im':'t1','fe':['k5_b25']}]
        if self.left or ((not self.left) and (not self.right)):
            side = []
            for a in self.radiomics_vox:
                im = a['im']
                for fe in a['fe']:
                    side.append(np.load('{}/{}/preloaded/{}/{}_radiomics_{}_left_{}.npy'.format(self.path,self.space,name,im,self.rad_vox_norm,fe))[:,self.feature_mask_vox])
            for inp in self.inps:
                side.append(np.load('{}/{}/preloaded/{}/{}_left.npy'.format(self.path,self.space,name,inp)))
            raw.append(np.concatenate(side,-1))
        if self.right or ((not self.left) and (not self.right)):
            side = []
            for a in self.radiomics_vox:
                im = a['im']
                for fe in a['fe']:
                    side.append(np.load('{}/{}/preloaded/{}/{}_radiomics_{}_right_{}.npy'.format(self.path,self.space,name,im,self.rad_vox_norm,fe))[:,self.feature_mask_vox])
            for inp in self.inps:
                side.append(np.load('{}/{}/preloaded/{}/{}_right.npy'.format(self.path,self.space,name,inp)))
            raw.append(np.concatenate(side,-1))
        raw = np.concatenate(raw,0)
        if self.extras is not None:
            raw = np.concatenate([raw,self.extras[name]],-1)
        return raw

    def getCon(self, name):
        raw = []
        if self.left or ((not self.left) and (not self.right)):
            raw.append(np.load('{}/{}/preloaded/{}/{}_left.npy'.format(self.path,self.space,name,self.outp)))
        if self.right or ((not self.left) and (not self.right)):
            raw.append(np.load('{}/{}/preloaded/{}/{}_right.npy'.format(self.path,self.space,name,self.outp)))
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
            if self.collapse_bin:
                raw = np.where(raw == 0, 0, 1)
        return np.array(raw,np.float16)

    def getOth(self, name):
        raw = []
        #[{'sp':'native','im':'t1','fe':['b10','b25','b50','b75'],'fi':['targets','roi','brain']}]
        shape_included = []
        #loop through spaces and images combos
        for a in self.radiomics:
            sp = a['sp']
            im = a['im']
            #loop through files per spaces/images
            for fi in a['fi']:
                #loop through bins per spaces/images
                for fe in a['fe']:
                    #only include shape once per file as they are constant, also shape should not matter in normalized space
                    if (sp == 'normalized') or (fi in shape_included):
                        mask = self.feature_mask_shapeless
                    else:
                        mask = self.feature_mask
                        shape_included.append(fi)
                    part = np.load('{}/{}/preloaded/{}/{}_radiomics_scale_{}_{}.npy'.format(self.path,sp,name,im,fe,fi))[:,mask]
                    part = self.getHemispheres(part, 0)
                    if self.single is not None and fi == 'targets' and not self.targets_all:
                        raw = raw[self.single:self.single+1,:]
                    raw.append(part.flatten())
        if len(self.features_clin) > 0:
            raw.append(np.load(self.path+'/'+self.space+'/preloaded/'+name+'/clinical.npy')[self.feature_mask_clin])
        raw = np.concatenate(raw)
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
        missing = pickleLoad(self.path+'/preprocessed/missing.pkl')
        t1t2 = False
        normalized = False
        if self.space == 'normalized':
            normalized = True
        if 't1t2' in self.inps:
            t1t2 =True
        for r in self.radiomics:
            if r['im'] == 't1t2':
                t1t2 =True
            if r['sp'] == 'normalized':
                normalized =True
        for r in self.radiomics_vox:
            if r['im'] == 't1t2':
                t1t2 =True
        if normalized:
            names = [n for n in names if n not in missing['normalized']]
        if t1t2:
            names = [n for n in names if n not in missing['t1t2']]
        if len(self.features_clin) > 0:
            names = [n for n in names if n not in missing['clinical']]
        for ex in self.exclude:
            names = [n for n in names if n not in missing[ex]]
        names = [n for n in names if self.outp not in missing.keys() or n not in missing[self.outp]]
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

