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
        type          = 'FFN',      # FNN CNN FCNN
        cnn_size      = 5,          #
        left          = True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
        right         = False,      #include right hemisphere data
        threshold     = 0.5,        #if float value provided, it thresholds the connectivty map
        binarize      = True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
        not_connected = True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
        single        = None,       #if int index value is provided, it only returns a specified connectivity map
        target        = False,      #
        roi           = False,      #
        brain         = False,      #
        features      = [],         #used radiomics features (emptylist means all)
        features_vox  = [],         #used voxel based radiomics features (emptylist means all)
        radiomics     = ['b25'],    #used radiomics features bin settings
        radiomics_vox = ['k5_b25'], #used voxel based radiomics features kernel and bin settings
        balance_data  = True,
        batch_size    = None,
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
        self.type = type
        self.cnn_size = cnn_size
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
        if self.type == 'FCNN':
            shapes = np.load(self.path+'/preprocessed/shapes.npy')
            self.shape = tuple(np.max(shapes,0))
        self.batch_size = batch_size

    def getData(self):
        if self.type == 'FCNN':
            return [self.getDatapoints(n) for n in self.names]
        return [self.getDatapoints(n) for n in self.names[0:2]]+[[self.getReconstructor(n) for n in self.names[2]]]
    
    def getReconstructor(self, name):
        x, y = self.getDatapoint(name, balance_override=True)
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
        return [x, y, lambda y:np.concatenate([self.reconstruct(y[:,i],idxs,bg.shape) for i in range(y.shape[-1])],-1), bg]

    def reconstruct(self, data, idxs, shape):
        ret = np.zeros(shape,np.float16)
        ret = ret.flatten()
        ret[idxs] = data
        ret = ret.reshape(shape)
        ret = np.expand_dims(ret,-1)
        return ret

    def getDatapoints(self, names):
        data = [self.getDatapoint(n) for n in names]
        x = [d[0] for d in data]
        y = [d[1] for d in data]
        if self.type == 'FCNN':
            x = np.array(x)
            y = np.array(y)
        elif self.type == 'CNN':
            x0 = [d[0] for d in x]
            x1 = [d[1] for d in x]
            x0 = np.concatenate(x0,0)
            if x1[0] is None:
                x1 = None
                x = [x0]
            else:
                x1 = np.concatenate(x1,0)
                x = [x0, x1]
            y = np.concatenate(y,0)
        elif self.type == 'FFN':
            x = np.concatenate(x,0)
            y = np.concatenate(y,0)
        if self.batch_size is not None:
            remainder = len(y) % self.batch_size
            print(remainder)
            if remainder > 0:
                y = np.concatenate([y,np.take(y,range(0,remainder),0)],0)
                if self.type == 'CNN':
                    for i in range(len(x)):
                        x[i] = np.concatenate([x[i],np.take(x[i],range(0,remainder),0)],0)
                else:
                    x = np.concatenate([x,np.take(x,range(0,remainder),0)],0)
        return [x, y]

    def getDatapoint(self, name, balance_override=False):
        if self.debug:
            print(name)
        x = self.getVox(name)
        y = self.getCon(name)
        if self.type == 'FCNN':
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
        elif self.type == 'CNN':
            mask = la.load(self.path+'/preprocessed/{}/roi.pkl'.format(name))
            mask_left = mask[:,:,:,0]
            mask_right = mask[:,:,:,1]
            mask_left = np.argwhere(mask_left)
            mask_right = np.argwhere(mask_right)
            idxs = []
            if self.left or ((not self.left) and (not self.right)):
                idxs.append(mask_left)
            if self.right or ((not self.left) and (not self.right)):
                idxs.append(mask_right)
            idxs = np.concatenate(idxs,0)
            bounds_l = idxs-(((self.cnn_size-1)//2)+self.cnn_size)
            bounds_u = bounds_l+self.cnn_size
            padded = np.zeros(tuple(np.array(x.shape[0:3])+(2*self.cnn_size))+(x.shape[3],),x.dtype)
            padded[self.cnn_size:-self.cnn_size,self.cnn_size:-self.cnn_size,self.cnn_size:-self.cnn_size,:] = x
            stacked = np.zeros((len(idxs),self.cnn_size,self.cnn_size,self.cnn_size,x.shape[3]),x.dtype)
            for i in range(len(idxs)):
                stacked[i,:,:,:,:] = padded[bounds_l[i,0]:bounds_u[i,0],bounds_l[i,1]:bounds_u[i,1],bounds_l[i,2]:bounds_u[i,2],:]
            x = stacked
            del padded
            del stacked
        if self.type in ['CNN','FFN']:
            if self.balance_data and not balance_override:
                dat = y
                if self.single is None and self.not_connected and self.threshold is not None and self.threshold >= 0.5:
                    dat = dat[:,0:-1]
                positive_idxs = np.argwhere(np.max(dat,1) >= 0.5).T[0]
                negative_cnt = len(y)-len(positive_idxs)
                for i in range(dat.shape[-1]):
                    positive_idxs = np.argwhere(dat[:,i] >= 0.5).T[0]
                    positive_cnt = len(positive_idxs)
                    remainder = negative_cnt % positive_cnt
                    positive_y = np.take(y,positive_idxs,0)
                    positive_x = np.take(x,positive_idxs,0)
                    y = [y,np.repeat(positive_y,negative_cnt//positive_cnt,0)]
                    x = [x,np.repeat(positive_x,negative_cnt//positive_cnt,0)]
                    if remainder > 0: y += [np.take(positive_y,range(0,remainder),0)]
                    if remainder > 0: x += [np.take(positive_x,range(0,remainder),0)]
                    y = np.concatenate(y,0)
                    x = np.concatenate(x,0)
            x1 = [x] if self.type == 'FFN' else []
            if self.target:
                x1.append(np.repeat(np.expand_dims(self.getOth(name,'targets').flatten(),0),len(x),0))
            if self.roi:
                x1.append(np.repeat(np.expand_dims(self.getOth(name,'roi').flatten(),0),len(x),0))
            if self.brain:
                x1.append(np.repeat(np.expand_dims(self.getOth(name,'t1_mask').flatten(),0),len(x),0))
            if len(x1) > 0:
                x1 = np.concatenate(x1,-1)
            else:
                x1 = None
            if self.type == 'CNN':
                return [[x,x1], y]
            if self.type == 'FFN':
                x = x1
        return [x, y]

    def getVox(self, name):
        if self.type in ['FCNN','CNN']:
            return np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_{}.npy'.format(self.path,name,rad))[:,:,:,self.feature_mask_vox] for rad in self.radiomics_vox],-1)
        else:
            raw = []
            if self.left or ((not self.left) and (not self.right)):
                raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_left_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
            if self.right or ((not self.left) and (not self.right)):
                raw.append(np.concatenate([np.load('{}/preloaded/{}/t1_radiomics_norm_right_{}.npy'.format(self.path,name,rad))[:,self.feature_mask_vox] for rad in self.radiomics_vox],-1))
            return np.concatenate(raw,0)

    def getCon(self, name):
        if self.type == 'FCNN':
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
            raw = self.getHemispheres(raw, idx=1)
            if self.single is not None:
                raw = raw[:,self.single]
        if self.threshold is not None:
            raw = np.where(raw <= self.threshold, 0, raw)
            if self.binarize:
                raw = convertToMask(raw)
        if self.single is None and self.not_connected and self.threshold is not None and self.threshold >= 0.5:
            if self.type == 'FCNN':
                mask = la.load(self.path+'/preprocessed/'+name+'/roi.pkl')
                mask = self.getHemispheres(mask)
            if raw.dtype == np.bool_:
                if self.type == 'FCNN':
                    nc = np.transpose(raw,[3,0,1,2])
                    nc = np.logical_or.reduce(nc)
                    nc = np.logical_xor(nc,mask)
                else:
                    nc = np.transpose(raw,[1,0])
                    nc = np.logical_or.reduce(nc)
                    nc = np.logical_not(nc)
            else:
                nc = (np.sum(raw, axis=-1)*-1)+1
                if self.type == 'FCNN':
                    nc = np.where(mask, nc, 0)
            nc = np.expand_dims(nc, -1)
            raw = np.concatenate([raw,nc],-1)
        return np.array(raw,np.float16)
    
    def getOth(self, name, file):
        if self.type == 'FCNN':
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
    
    def getHemispheres(self, data, idx=0):
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
            half = data.shape[idx]//2
            if self.left and not self.right:
                return data[:half,:] if idx == 0 else data[:,:half]
            if not self.left and self.right:
                return data[half:,:] if idx == 0 else data[:,half:]
            return data[:half,:]+data[half:,:] if idx == 0 else data[:,:half]+data[:,half:]
    
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
