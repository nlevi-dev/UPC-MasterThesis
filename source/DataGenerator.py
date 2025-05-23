import numpy as np
import LayeredArray as la
from util import convertToMask, pickleLoad

class DataGenerator():
    def __init__(self,
        path          = 'data',         #path of the data
        seed          = 42,             #random seed for the train/val/test splits
        split         = 0.8,            #train/all ratio
        test_split    = 0.5,            #test/(test+validation) ratio
        control       = True,           #include control records
        huntington    = False,          #include huntington records
        left          = False,          #include left hemisphere datapoints
        right         = False,          #include right hemisphere datapoints
        threshold     = 0.6,            #if not None it thresholds the labels by setting the values under the provided threshold to zero; if 0 it re-one-hot encodes the labels (sets the maximum label to 1 and the rest to 0)
        binarize      = True,           #if thresholded and True, it also sets the labels above the threshold to 1 (and the rest to 0)
        not_connected = True,           #if thresholded and True, it appends an additional 'not connected' label, which complements the sum of the labels per voxel to 1
        single        = None,           #if not None it only returns the label with the provided index
        features      = [],             #used radiomics features (emptylist means all)
        features_vox  = [],             #used voxel based radiomics features (emptylist means all)
        radiomics     = [               #non-voxel based space, input image, bin and file selection
            #{'sp':'native','im':'t1','fe':['b10','b25','b50','b75'],'fi':['targets','roi','t1_mask']},
        ],
        space         = 'native',       #voxel based space selection (native/normalized)
        radiomics_vox = [               #voxel based input image, kernel and bin selection
            {'im':'t1','fe':['k5_b25']},
        ],
        rad_vox_norm  = 'norm',         #use normalization, or only min-max scaling on the voxel based radiomic features (norm/scale)
        inps          = [],             #additional voxel inputs (t1/t1t2/diffusion/diffusion_fa/diffusion_md/diffusion_rd)
        features_clin = None,           #additional clinical data inputs (empty array means all)
        outp          = 'connectivity', #output (connectivity/streamline/basal_seg/diffusion_fa/diffusion_md/diffusion_rd)
        balance_data  = True,           #enables data balancing
        balance_bins  = 10,             #number of bins used for continuous data when balancing
        balance_ratio = 1,              #ratio of the resampling of the difference between each bin and the max bin when balancing (where 0 is unbalanced and 1 is perfectly balanced)
        exclude       = ['normalized'], #can manually add missing groups of records to exclude (t1t2/normalized/basal_seg/diffusion_fa)
        reinclude     = [],             #can manually re-include (append) missing groups of records to the train split (t1t2/normalized/basal_seg/diffusion_fa)
        debug         = False,          #only returns 1/1/1 records for train/val/test when True
        augment       = [],             #list of record suffixes, used to include augmented records in the training split; for example the suffix '_5_0_0' is used for the rotation augmentation
        include_warp  = False,          #DEPRECATED
        collapse_max  = False,          #UNUSED collapses the last dimesnion with maximum function (used for regression)
        collapse_bin  = False,          #UNUSED binarizes the collapsed output layer
        extras        = None,           #UNUSED includes extra data for each record (format {'datapoint_name':[data]})
        targets_all   = False,          #UNUSED includes all target regions regardless if single or not
        pca           = None,           #NOT IMPLEMENTED if provided a float value it keeps that fraction of the explained variance
        pca_parts     = None,           #NOT IMPLEMENTED only applies PCA to parts of the input space, possible values: [vox,target,roi,brain]
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
        self.reinclude = reinclude
        self.augment = augment
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
        self.balance_bins = balance_bins
        self.balance_ratio = balance_ratio
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
        return [self.getDatapoints(n) for n in self.names[:cnt]]
    
    def getReconstructor(self, name, xy_only=False):
        x, y = self.getDatapoint(name)
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

        splits1 = [0]
        for p in y:
            splits1.append(splits1[-1]+len(p))
        splits1 = np.array(splits1,np.uint32)

        warp = []
        cnt = 0
        splits2 = [0]
        for i in range(len(names)):
            c = 0
            if self.left or ((not self.left) and (not self.right)):
                conv = np.load('{}/{}/preloaded/{}/{}_left.npy'.format(self.path,self.space,names[i],'nat2norm' if self.space == 'native' else 'norm2nat'))+cnt
                warp.append(conv)
                cnt += len(np.load('{}/{}/preloaded/{}/coords_left.npy'.format(self.path,self.space,names[i])))
                c += len(conv)
            if self.right or ((not self.left) and (not self.right)):
                conv = np.load('{}/{}/preloaded/{}/{}_right.npy'.format(self.path,self.space,names[i],'nat2norm' if self.space == 'native' else 'norm2nat'))+cnt
                warp.append(conv)
                cnt += len(np.load('{}/{}/preloaded/{}/coords_right.npy'.format(self.path,self.space,names[i])))
                c += len(conv)
            splits2.append(splits2[-1]+c)
        splits2 = np.array(splits2,np.uint32)

        warp = np.concatenate(warp,0)

        x = np.concatenate(x,0)
        y = np.concatenate(y,0)
        
        if self.balance_data:
            if y.shape[1] == 1:
                dat = y[:,0]
                bins, edges = np.histogram(dat,self.balance_bins)
                bin_masks = np.array([np.logical_and(edges[i] <= dat,dat < edges[i+1]) for i in range(len(bins))])
                bins = np.count_nonzero(bin_masks,1)
                maxbin = np.max(bins)
            else:
                dat = np.argmax(y,1)
                bin_masks = np.array([(dat == i) for i in range(y.shape[-1])])
                bins = np.count_nonzero(bin_masks,1)
                maxbin = np.max(bins)
            yc = [y]
            xc = [x]
            for i in range(len(bins)):
                positive_y = y[bin_masks[i],:]
                positive_x = x[bin_masks[i],:]
                bincnt = bins[i]
                if bincnt == 0:
                    continue
                remainder = maxbin % bincnt
                remainder = int(remainder * self.balance_ratio)
                div = maxbin // bincnt - 1
                div = div * self.balance_ratio
                remainder2 = int(bincnt*(div-int(div)))
                div = int(div)
                if (remainder == 0 and div == 0):
                    continue
                if div > 0:
                    yc.append(np.repeat(positive_y,div,0))
                    xc.append(np.repeat(positive_x,div,0))
                if remainder > 0:
                    yc.append(positive_y[:remainder,:])
                    xc.append(positive_x[:remainder,:])
                if remainder2 > 0:
                    yc.append(positive_y[-remainder2:,:])
                    xc.append(positive_x[-remainder2:,:])
            y = np.concatenate(yc,0)
            x = np.concatenate(xc,0)
        
        return [x, y, warp, [splits1, splits2]]

    def getDatapoint(self, name):
        x = self.getVox(name)
        y = self.getCon(name)
        x1 = [x]
        if len(self.radiomics) > 0 or len(self.features_clin) > 0:
            app = np.repeat(np.expand_dims(self.getOth(name),0),len(x),0)
            x1.append(app)
        x = np.concatenate(x1,-1)
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
        asymptomatic = np.load(self.path+'/asymptomatic.npy')
        missing = pickleLoad(self.path+'/preprocessed/missing.pkl')
        t1t2 = False
        normalized = False
        if self.space == 'normalized' or 'coords' in self.inps:
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
        asym = [n for n in names if n[0] == 'H' and n in asymptomatic]
        huns = [n for n in names if n[0] == 'H' and n not in asymptomatic]
        ran = np.random.default_rng(self.seed)
        ran.shuffle(cons)
        ran.shuffle(asym)
        ran.shuffle(huns)
        s0 = self.split
        s1 = self.split+(1-self.split)*self.test_split
        cs0 = int(len(cons)*s0)
        cs1 = int(len(cons)*s1)
        if cs1 > len(cons)-1:
            cs1 = len(cons)-1
        if cs0 >= cs1:
            cs0 = cs1-1
        as0 = int(len(asym)*s0)
        as1 = int(len(asym)*s1)
        if as1 > len(asym)-1:
            as1 = len(asym)-1
        if as0 >= as1:
            as0 = as1-1
        hs0 = int(len(huns)*s0)
        hs1 = int(len(huns)*s1)
        if hs1 > len(huns)-1:
            hs1 = len(huns)-1
        if hs0 >= hs1:
            hs0 = hs1-1
        cons_train = cons[:cs0]
        cons_test  = cons[cs0:cs1]
        cons_val   = cons[cs1:]
        asym_train = asym[:as0]
        asym_test  = asym[as0:as1]
        asym_val   = asym[as1:]
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
            tr = tr+asym_train+huns_train
            te = te+asym_test+huns_test
            va = va+asym_val+huns_val
        ran.shuffle(tr)
        ran.shuffle(te)
        ran.shuffle(va)
        for ic in self.reinclude:
            additional = [n for n in missing[ic] if (n not in tr) and (n not in te) and (n not in va)]
            if self.control:
                tr += [n for n in additional if n[0] == 'C']
            if self.huntington:
                tr += [n for n in additional if n[0] == 'H']
        for rot in self.augment:
            tr += [n+rot for n in tr]
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

