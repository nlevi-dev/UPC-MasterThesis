import multiprocessing
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
        normalize     = True,       #if true it normalizes some of the features with log10
        threshold     = None,       #if float value provided, it thresholds the connectivty map
        binarize      = False,      #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
        not_connected = True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
        single        = None,       #if int index value is provided, it only returns a specified connectivity map
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
        self.normalize = normalize
        self.threshold = False
        self.threshold_val = -1
        if threshold is not None:
            self.threshold_val = threshold
            self.threshold = True
        self.binarize = binarize
        self.not_connected = not_connected
        self.single = False
        self.single_val = -1.0
        if single is not None:
            self.single_val = single
            self.single = True
        self.target = target
        self.roi = roi
        self.brain = brain
        self.feature_idxs = getFeatureIdxs(features,np.load(path+'/preprocessed/features.npy'))
        self.feature_idxs_vox = getFeatureIdxs(features_vox,np.load(path+'/preprocessed/features_vox.npy'))
        self.radiomics = radiomics
        self.radiomics_vox = radiomics_vox
        labels = np.load(self.path+'/preprocessed/labels.npy')
        if self.single:
            l = 1
        else:
            l = len(labels)*2
            if (not self.left) or (not self.right):
                l = l//2
            if self.not_connected and self.threshold and self.threshold_val >= 0.5:
                l = l+1
        #precompute spaital shape or non spatial lengths
        if self.spatial:
            shapes = np.load(self.path+'/preprocessed/shapes.npy')
            self.shape = tuple(np.max(shapes,0))
            self.length = len(self.names)//self.batch_size
            self.x_shape = self.shape+(len(self.feature_idxs_vox)*len(self.radiomics_vox),)
            self.y_shape = self.shape+(l,)
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
                if len(self.mask_lengths) > 0:
                    mask_cnt += self.mask_lengths[-1]
                self.mask_lengths.append(mask_cnt)
            self.length = self.mask_lengths[-1]//self.batch_size
            self.x_shape = (len(self.feature_idxs_vox)*len(self.radiomics_vox),)
            self.y_shape = (l,)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        olo = self.batch_size*idx
        ohi = olo+self.batch_size
        if self.spatial:
            lo = olo
            hi = ohi
        else:
            l = 0
            h = 0
            for i in range(len(self.mask_lengths)+1):
                bl = self.mask_lengths[i-1] if i > 0 else 0
                bu = self.mask_lengths[i]
                if bl <= olo and olo <= bu:
                    l = i
                if bl <= ohi and ohi <= bu:
                    h = i
                    break
            lo = l
            hi = h+1
        vox = [] #(d,x,y,z,f) || (d,p,f)
        con = [] #(d,x,y,z,c) || (d,p,c)
        tar = [] #(d,x,y,z,c) || (d,c,f)
        roi = [] #(d,x,y,z,r) || (d,r,f)
        bra = [] #(d,x,y,z)   || (d,f)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            res = pool.map(processDatapointWrapper, [[self,name] for name in self.names[lo:hi]])
        for r in res:
            vox.append(r[0])
            con.append(r[1])
            tar.append(r[2])
            roi.append(r[3])
            bra.append(r[4])
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
            shift = self.mask_lengths[lo-1] if lo > 0 else 0
            x = x[olo-shift:ohi-shift,:]
            y = y[olo-shift:ohi-shift,:]
        assert (self.batch_size,)+self.x_shape == x.shape
        assert (self.batch_size,)+self.y_shape == y.shape
        return [x, y]

def processDatapointWrapper(inp):
    self, name = inp
    return processDatapoint(self, name)

def processDatapoint(self:DataGenerator, name:str):
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
        center = None
        mask = mask.flatten()
    mask_cnt = np.count_nonzero(mask)
    #load voxel based radiomic features
    vox = getVox(self, name, center, mask, mask_cnt)
    #load connectivity maps
    con = getCon(self, name, center, mask, mask_cnt)
    #load cortical targets
    if self.target:
        tar = getOth(self, name, center, 'targets')
    else:
        tar = None
    #load roi
    if self.roi:
        roi = getOth(self, name, center, 'roi')
    else:
        roi = None
    #load brain
    if self.brain:
        bra = getOth(self, name, center, 't1_mask')
    else:
        bra = None
    return [vox,con,tar,roi,bra]

def getVox(self:DataGenerator, name:str, center:np.ndarray, mask:np.ndarray, mask_cnt:int):
    ret = []
    #for each radiomics setting (kernelWidth/binWidth pairs)
    for rad in self.radiomics_vox:
        #load raw
        raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'.npy',mmap_mode='r')
        factors = np.load(self.path+'/preprocessed/features_scale_vox_'+rad+'.npy')
        #get output shape
        s = self.shape if self.spatial else (mask_cnt,)
        s = s+(len(self.feature_idxs_vox),)
        res = np.zeros(s, np.float16)
        #for each feature
        for j in range(len(self.feature_idxs_vox)):
            #get feature slice
            f = self.feature_idxs_vox[j]
            slc = raw[:,:,:,f]
            #flatten
            if not self.spatial:
                slc = slc.flatten()[mask]
            #normalize
            if self.normalize and factors[f][2] == 'log10':
                slc = np.log10(slc+1)
                fac = np.array(factors[f][3:5],slc.dtype)
            else:
                fac = np.array(factors[f][0:2],slc.dtype)
            #scale
            slc = (slc-fac[0])/(fac[1]-fac[0])
            #insert into resolution
            if self.spatial:
                res[center[0]:center[0]+slc.shape[0],
                    center[1]:center[1]+slc.shape[1],
                    center[2]:center[2]+slc.shape[2],j] = slc
            else:
                res[:,j] = slc
        ret.append(res)
    return np.concatenate(ret,-1)

def getCon(self:DataGenerator, name:str, center:np.ndarray, mask:np.ndarray, mask_cnt:int):
    #load raw
    raw = la.load(self.path+'/preprocessed/'+name+'/connectivity.pkl')
    raw = getHemispheres(raw,self.left,self.right)
    #get output shape
    s = self.shape if self.spatial else (mask_cnt,)
    s = s+(1 if self.single else raw.shape[-1],)
    con = np.zeros(s, np.bool_ if (self.threshold and self.binarize) else np.float16)
    #for each connectivity map
    for j in range(raw.shape[-1]):
        #only process single
        if self.single:
            if j != self.single_val:
                continue
        #get connectivity slice
        slc = raw[:,:,:,j]
        #flatten
        if not self.spatial:
            slc = slc.flatten()[mask]
        #threshold
        if self.threshold:
            slc = np.where(slc <= self.threshold_val, 0, slc)
            #binarize
            if self.binarize:
                slc = convertToMask(slc)
        if self.single:
            j = 0
        #insert into resolution
        if self.spatial:
            con[center[0]:center[0]+slc.shape[0],
                center[1]:center[1]+slc.shape[1],
                center[2]:center[2]+slc.shape[2],j] = slc
        else:
            con[:,j] = slc
        if self.single:
            break
    #compute not connected layer
    if not self.single and self.not_connected and self.threshold_val >= 0.5:
        if con.dtype == np.bool_:
            if self.spatial:
                nc = np.transpose(con,[3,0,1,2])
                nc = np.logical_or.reduce(nc)
                nc = np.logical_xor(nc,mask)
            else:
                nc = np.transpose(con,[1,0])
                nc = np.logical_or.reduce(nc)
                nc = np.logical_not(nc)
        else:
            nc = (np.sum(con, axis=-1)*-1)+1
            if self.spatial:
                nc = np.where(mask, nc, 0)
        nc = np.expand_dims(nc, -1)
        con = np.concatenate([con,nc],-1)
    return con

def getOth(self:DataGenerator, name:str, center:np.ndarray, file:str):
    if self.spatial:
        if file in ['targets','roi']:
            raw = la.load(self.path+'/preprocessed/'+name+'/'+file+'.pkl')
            raw = getHemispheres(raw,self.left,self.right)
            ret = np.zeros(self.shape+(raw.shape[3],),raw.dtype)
            ret[center[0]:center[0]+raw.shape[0],
                center[1]:center[1]+raw.shape[1],
                center[2]:center[2]+raw.shape[2],:] = raw
        else:
            raw = np.load(self.path+'/preprocessed/'+name+'/'+file+'.npy')
            ret = np.zeros(self.shape+(1,),raw.dtype)
            ret[center[0]:center[0]+raw.shape[0],
                center[1]:center[1]+raw.shape[1],
                center[2]:center[2]+raw.shape[2],0] = raw
    else:
        ret = []
        for rad in self.radiomics:
            raw = np.load(self.path+'/preprocessed/'+name+'/t1_radiomics_raw_'+rad+'_'+file+'.npy')
            if len(raw.shape) == 1:
                raw = np.expand_dims(raw,0)
            factors = np.load(self.path+'/preprocessed/features_scale_'+rad+'.npy')
            res = np.zeros((raw.shape[0],len(self.feature_idxs)), np.float16)
            for j in range(len(self.feature_idxs)):
                f = self.feature_idxs[j]
                slc = raw[:,f]
                factor = factors[f]
                slc = (slc-factor[0])/(factor[1]-factor[0])
                res[:,f] = slc
            if res.shape[0] > 1:
                res = getHemispheres(res,self.left,self.right)
            ret.append(res)
        ret = np.concatenate(ret,-1)
    return ret

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

def getFeatureIdxs(features, raw_features):
    if features is None or len(features) == 0:
        features = raw_features
    feature_idxs = []
    for i in range(len(raw_features)):
        if raw_features[i] in features:
            feature_idxs.append(i)
    feature_idxs = np.array(feature_idxs)
    return feature_idxs

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
