import math
import numpy as np
import keras
import LayeredArray as la
from util import convertToMask

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
        path          = 'data',     #path of the data
        seed          = 42,         #seed for the split
        split         = 0.8,        #train/all ratio
        train         = True,       #training/testing split
        control       = True,       #include control data points
        huntington    = True,       #include huntington data points
        batch_size    = 4,          #batch size
        spatial       = False,      #keep spaital format of flatten voxels in the brain region
        left          = True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = True,       #include right hemisphere data
        threshold     = False,      #if float value provided, it thresholds the connectivty map
        binarize      = False,      #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
        not_connected = True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
        single        = False,      #if int index value is provided, it only returns a specified connectivity map
        target        = False,
        roi           = False,
        brain         = False,
        features      = [],         #used radiomics features (emptylist means all)
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics     = ['b25'],    #used radiomics features bin settings
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
    ):
        self.path = path
        self.names = getSplit(path,seed,split,train,control,huntington)
        self.batch_size = batch_size
        self.spatial = spatial
        self.left = left
        self.right = right
        self.threshold = False
        self.threshold_val = -1
        if threshold is not None and threshold != False:
            self.threshold_val = threshold
            self.threshold = True
        self.binarize = binarize
        self.not_connected = not_connected
        self.single = False
        self.single_val = -1.0
        if single is not None and single != False:
            self.single_val = single
            self.single = True
        self.target = target
        self.roi = roi
        self.brain = brain
        self.feature_idxs, self.feature_idxs_vox = getFeatureIdxs(path,features,features_vox)
        self.radiomics = radiomics
        self.radiomics_vox = radiomics_vox
        #precompute spaital shape or non spatial lengths
        if self.spatial:
            shapes = np.load(self.path+'/preprocessed/shapes.npy')
            self.shape = tuple(np.max(shapes,0))
        else:
            self.mask_lengths = []
            for name in self.names:
                mask = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
                le = self.left
                ri = self.right
                if le and ri:
                    le = False
                    ri = False
                mask = getHemispheres(mask, le, ri)
                mask = mask.flatten()
                mask_cnt = np.count_nonzero(mask)
                self.mask_lengths.append(mask_cnt)

    def __len__(self):
        return math.ceil(len(self.names)/self.batch_size)

    def getitem(self, idx):
        lo = self.batch_size*idx
        hi = lo+self.batch_size
        if hi > len(self.names): hi = len(self.names)
        vox = [] #(d,x,y,z,f) || (d,p,f)
        con = [] #(d,x,y,z,c) || (d,p,c)
        tar = [] #(d,x,y,z,c) || (d,c,f)
        roi = [] #(d,x,y,z,r) || (d,r,f)
        bra = [] #(d,x,y,z)   || (d,f)
        for i in range(lo,hi):
            name = self.names[i]
            #load mask
            mask = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
            le = self.left
            ri = self.right
            if le and ri:
                le = False
                ri = False
            mask = getHemispheres(mask, le, ri)
            mask = mask[:,:,:,0]
            if self.spatial:
                center = (np.array(self.shape)-np.array(mask.shape))//2
                m = np.zeros(self.shape,mask.dtype)
                m[center[0]:center[0]+mask.shape[0],
                  center[1]:center[1]+mask.shape[1],
                  center[2]:center[2]+mask.shape[2]] = mask
                mask = m
            else:
                mask = mask.flatten()
            mask_cnt = np.count_nonzero(mask)
            #load voxel based radiomic features
            voxs = []
            for rad in self.radiomics_vox:
                # raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'.npy')
                # factors = np.load(self.path+'/preprocessed/features_scale_vox_'+rad+'.npy')
                raw = np.repeat(np.expand_dims(np.load(self.path+'/preprocessed/'+name+'/t1.npy'),-1),len(self.feature_idxs_vox),-1)
                factors = np.array([[0,1,''] for _ in range(len(self.feature_idxs_vox))])
                #==============================================================================
                s = self.shape if self.spatial else (mask_cnt,)
                s = s+(len(self.feature_idxs_vox),)
                res = np.zeros(s, np.float16)
                for j in range(len(self.feature_idxs_vox)):
                    f = self.feature_idxs_vox[j]
                    slice = raw[:,:,:,f] if self.spatial else raw[:,:,:,f].flatten()[mask]
                    factor = np.array(factors[f][0:2],slice.dtype)
                    slice = (slice-factor[0])/(factor[1]-factor[0])
                    if factors[f][2] == 'log10':
                        slice = np.log10(slice)
                    if self.spatial:
                        res[center[0]:center[0]+slice.shape[0],
                            center[1]:center[1]+slice.shape[1],
                            center[2]:center[2]+slice.shape[2],j] = slice
                    else:
                        res[:,j] = slice
                voxs.append(res)
            voxs = np.concatenate(voxs,-1)
            vox.append(voxs)
            #load connectivity maps
            raw = la.load(self.path+'/preprocessed/'+name+'/connectivity.pkl')
            raw = getHemispheres(raw,self.left,self.right)
            s = self.shape if self.spatial else (mask_cnt,)
            s = s+(raw.shape[-1],)
            res = np.zeros(s, np.bool_ if self.threshold else np.float16)
            for j in range(raw.shape[-1]):
                slice = raw[:,:,:,j] if self.spatial else raw[:,:,:,j].flatten()[mask]
                if self.threshold:
                    slice = np.where(slice <= self.threshold_val, 0, slice)
                    if self.binarize:
                        slice = convertToMask(slice)
                if self.spatial:
                    res[center[0]:center[0]+slice.shape[0],
                        center[1]:center[1]+slice.shape[1],
                        center[2]:center[2]+slice.shape[2],j] = slice
                else:
                    res[:,j] = slice
            if self.single:
                res = np.take(res,self.single_val,-1)
            elif self.not_connected and self.threshold_val >= 0.5:
                if res.dtype == np.bool_:
                    if self.spatial:
                        nc = np.transpose(res,[3,0,1,2])
                        nc = np.logical_or.reduce(nc)
                        nc = np.logical_xor(nc,mask)
                    else:
                        nc = np.transpose(res,[1,0])
                        nc = np.logical_or.reduce(nc)
                        nc = np.logical_not(nc)
                else:
                    nc = (np.sum(x, axis=-1)*-1)+1
                    if self.spatial:
                        nc = np.where(mask, nc, 0)
                nc = np.expand_dims(nc, -1)
                res = np.concatenate([res,nc],-1)
            con.append(res)
            #load cortical targets
            if self.target:
                if self.spatial:
                    raw = la.load(self.path+'/preprocessed/'+name+'/targets.pkl')
                    raw = getHemispheres(raw,self.left,self.right)
                    r = np.zeros(self.shape+(raw.shape[3],),raw.dtype)
                    r[center[0]:center[0]+raw.shape[0],
                      center[1]:center[1]+raw.shape[1],
                      center[2]:center[2]+raw.shape[2],:] = raw
                    raw = r
                    tar.append(raw)
                else:
                    tars = []
                    for rad in self.radiomics:
                        raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'_targets.npy')
                        factors = np.load(self.path+'/preprocessed/features_scale_'+rad+'.npy')
                        res = np.zeros((raw.shape[0],len(self.feature_idxs)), np.float16)
                        for j in range(len(self.feature_idxs)):
                            f = self.feature_idxs[j]
                            slice = raw[:,f]
                            factor = factors[f]
                            slice = (slice-factor[0])/(factor[1]-factor[0])
                            res[:,f] = slice
                        res = getHemispheres(res,self.left,self.right)
                        tars.append(res)
                    tars = np.concatenate(tars,-1)
                    tar.append(tars)
            #load roi
            if self.roi:
                if self.spatial:
                    raw = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
                    raw = getHemispheres(raw,self.left,self.right)
                    r = np.zeros(self.shape+(raw.shape[3],),raw.dtype)
                    r[center[0]:center[0]+raw.shape[0],
                      center[1]:center[1]+raw.shape[1],
                      center[2]:center[2]+raw.shape[2],:] = raw
                    raw = r     
                    roi.append(raw)
                else:
                    rois = []
                    for rad in self.radiomics:
                        raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'_roi.npy')
                        factors = np.load(self.path+'/preprocessed/features_scale_'+rad+'.npy')
                        res = np.zeros((raw.shape[0],len(self.feature_idxs)), np.float16)
                        for j in range(len(self.feature_idxs)):
                            f = self.feature_idxs[j]
                            slice = raw[:,f]
                            factor = factors[f]
                            slice = (slice-factor[0])/(factor[1]-factor[0])
                            res[:,f] = slice
                        res = getHemispheres(res,self.left,self.right)
                        rois.append(res)
                    rois = np.concatenate(rois,-1)
                    roi.append(rois)
            #load brain
            if self.brain:
                if self.spatial:
                    raw = np.load(self.path+'/preprocessed/'+name+'/t1_mask.npy')
                    r = np.zeros(self.shape,raw.dtype)
                    r[center[0]:center[0]+raw.shape[0],
                      center[1]:center[1]+raw.shape[1],
                      center[2]:center[2]+raw.shape[2]] = raw
                    raw = r    
                    bra.append(raw)
                else:
                    bras = []
                    for rad in self.radiomics:
                        raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'_t1_mask.npy')
                        raw = np.expand_dims(raw,0)
                        factors = np.load(self.path+'/preprocessed/features_scale_'+rad+'.npy')
                        res = np.zeros((raw.shape[0],len(self.feature_idxs)), np.float16)
                        for j in range(len(self.feature_idxs)):
                            f = self.feature_idxs[j]
                            slice = raw[:,f]
                            factor = factors[f]
                            slice = (slice-factor[0])/(factor[1]-factor[0])
                            res[:,f] = slice
                        bras.append(res)
                    bras = np.concatenate(bras,-1)
                    bra.append(bras[0])
        #assemble data
        #vox (d,x,y,z,f) || (d,p,f)
        #con (d,x,y,z,c) || (d,p,c) => if sinlge (d,x,y,z) || (d,p) => if not_connected (d,x,y,z,c+1) || (d,p,c+1)
        #tar (d,x,y,z,c) || (d,c,f)
        #roi (d,x,y,z,r) || (d,r,f)
        #bra (d,x,y,z)   || (d,f)
        if self.spatial:
            x = np.array(vox)
            y = np.array(con)
        else:
            x = np.concatenate(vox,0)
            y = np.concatenate(con,0)
        return [x, y]

    def __getitem__(self, idx):
        self.getitem(idx)

def getSplit(path, seed, split, train, control, huntington):
    if not control and not huntington:
        raise Exception('Must include control and/or huntington data points!')
    names = np.load(path+'/preprocessed/names.npy')
    cons = [n for n in names if n[0] == 'C']
    huns = [n for n in names if n[0] == 'H']
    ran = np.random.default_rng(seed)
    ran.shuffle(cons)
    ran.shuffle(huns)
    cons_train = cons[:int(len(cons)*split)]
    cons_test  = cons[int(len(cons)*split):]
    huns_train = cons[:int(len(cons)*split)]
    huns_test  = cons[int(len(cons)*split):]
    tr = []
    te = []
    if control:
        tr = tr+cons_train
        te = te+cons_test
    if huntington:
        tr = tr+huns_train
        te = te+huns_test
    ran.shuffle(tr)
    ran.shuffle(te)
    return tr if train else te

def getFeatureIdxs(path,features,features_vox):
    raw_features = np.load(path+'/preprocessed/features.npy')
    raw_features_vox = np.load(path+'/preprocessed/features.npy')[0:92]#np.load(path+'/preprocessed/features_vox.npy')
    
    if features is None or len(features) == 0:
        features = raw_features
    feature_idxs = []
    for i in range(len(raw_features)):
        if raw_features[i] in features:
            feature_idxs.append(i)
    feature_idxs = np.array(feature_idxs)

    if features_vox is None or len(features_vox) == 0:
        features_vox = raw_features_vox
    feature_idxs_vox = []
    for i in range(len(raw_features_vox)):
        if raw_features_vox[i] in features_vox:
            feature_idxs_vox.append(i)
    feature_idxs_vox = np.array(feature_idxs_vox)

    return [feature_idxs,feature_idxs_vox]

def getHemispheres(data, left, right):
    if left and right:
        return data
    if len(data.shape) == 4:
        half = data.shape[-1]//2
        if left:
            return data[:,:,:,:half]
        if right:
            return data[:,:,:,half:]
        if data.dtype == np.bool_:
            return np.logical_or(data[:,:,:,:half],data[:,:,:,half:])
        return data[:,:,:,:half]+data[:,:,:,half:]
    half = data.shape[0]//2
    if left:
        return data[:half,:]
    if right:
        return data[half:,:]
    return np.concatenate([data[:half,:],data[half:,:]],1)