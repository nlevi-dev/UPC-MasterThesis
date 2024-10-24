import numpy as np
import _pickle as pickle
from util import convertToMask, findMaskBounds

def save(path, data):
    d = LayeredArray(data)
    d.save(path)

def load(path):
    return LayeredArray.load(path).getData()

class LayeredArray:
    def __init__(self, data):
        self.setData(data)

    def setData(self, data):
        #get shape and dtype
        if len(data.shape) != 4: raise Exception('4d shape expected!')
        self.shape = np.array(data.shape,dtype=np.int16)
        self.dtype = data.dtype
        #compute each layer
        self.bounds = []
        self.data = []
        if self.dtype == np.bool_:
            mask = data
        else:
            mask = convertToMask(data)
        for i in range(self.shape[3]):
            b = findMaskBounds(mask[:,:,:,i])
            d = np.array(data[b[0,0]:b[0,1],b[1,0]:b[1,1],b[2,0]:b[2,1],i],self.dtype)
            self.bounds.append(b)
            self.data.append(d)

    def getData(self):
        ret = np.zeros(self.shape, dtype=self.dtype)
        for i in range(self.shape[3]):
            ret[self.bounds[i][0,0]:self.bounds[i][0,1],
                self.bounds[i][1,0]:self.bounds[i][1,1],
                self.bounds[i][2,0]:self.bounds[i][2,1],i] = self.data[i]
        return ret
    
    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            ret = pickle.load(f)
        return ret

    def save(self, path):
        with open(path+'.pkl','wb') as f:
            pickle.dump(self,f)