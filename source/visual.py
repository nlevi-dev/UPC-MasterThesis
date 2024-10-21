import multiprocessing
import colorsys
import numpy as np
try:
    tmp = get_ipython().__class__.__name__
    if tmp == 'ZMQInteractiveShell':
        import matplotlib.pyplot as plt
    else:
        raise Exception('err')
except:
    from matplotlib_terminal import plt
from util import convertToMask

def thresholdArray(data, threshold):
    mask = np.zeros(data.shape,dtype=np.bool_)
    mask = np.where(data <= threshold, False, True)
    return mask

def colorChannel(data, channel=0):
    if data.dtype != np.bool_:
        mi = np.min(data)
        ma = np.max(data)
        data = (data-mi)/(ma-mi)
    ret = np.zeros(data.shape+(3,),dtype=data.dtype)
    ret[:,:,channel] = data
    return ret

def formatTime(s):
    m = s//60
    h = m//60
    if h > 0:
        return '{}h{}m{}s'.format(round(h),round(m-h*60),round(s-m*60))
    if m > 0:
        return '{}m{}s'.format(round(m),round(s-m*60))
    return '{}s'.format(round(s))

def formatThreadName():
    thr = multiprocessing.current_process().name
    if thr == 'MainProcess':
        return 'main'
    thr = thr[thr.index("-")+1:]
    return thr

def pickColors(count):
    return [colorsys.hsv_to_rgb(c*(1.0/count),1,1) for c in range(count)]

def showSlices(data1, data2=None, title='', color=False):
    slices = [data1.shape[0]//2,data1.shape[1]//2,2*data1.shape[2]//3]
    if data2 is not None and len(data2.shape) == 4:
        colors = pickColors(data2.shape[3])
        if data1.dtype != np.bool_:
            data1 = convertToMask(data1)
        if data2.dtype != np.bool_:
            data2 = np.array(thresholdArray(data2, 0.5),dtype=np.float16)
        data1 = np.array(data1,dtype=np.float16)-np.array(convertToMask(np.sum(data2,3)),dtype=np.float16)
        d0 = np.repeat(np.expand_dims(data1[slices[0],:,:],-1),3,-1)
        d1 = np.repeat(np.expand_dims(data1[:,slices[1],:],-1),3,-1)
        d2 = np.repeat(np.expand_dims(data1[:,:,slices[2]],-1),3,-1)
        d0 = np.where(d0 < 0, 0, d0)
        d1 = np.where(d1 < 0, 0, d1)
        d2 = np.where(d2 < 0, 0, d2)
        for i in range(len(colors)):
            c0 = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(colors[i],dtype=np.float16),0),d0.shape[1],0),0),d0.shape[0],0)
            c1 = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(colors[i],dtype=np.float16),0),d1.shape[1],0),0),d1.shape[0],0)
            c2 = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.array(colors[i],dtype=np.float16),0),d2.shape[1],0),0),d2.shape[0],0)
            d0 = d0+(c0*np.repeat(np.expand_dims(data2[slices[0],:,:,i],-1),3,-1))
            d1 = d1+(c1*np.repeat(np.expand_dims(data2[:,slices[1],:,i],-1),3,-1))
            d2 = d2+(c2*np.repeat(np.expand_dims(data2[:,:,slices[2],i],-1),3,-1))
        d0 = np.where(d0 > 1, 1, d0)
        d1 = np.where(d1 > 1, 1, d1)
        d2 = np.where(d2 > 1, 1, d2)
    else:
        if data2 is not None: color=True
        c = (lambda x:colorChannel(x,0)) if color else (lambda x:x)
        d0 = c(data1[slices[0],:,:])
        d1 = c(data1[:,slices[1],:])
        d2 = c(data1[:,:,slices[2]])
        if data2 is not None:
            if len(data2.shape) == 5:
                data2 = data2[0]
            if len(data2.shape) == 4:
                data2 = data2[:,:,:,0]
            d0 = d0 + colorChannel(data2[slices[0],:,:],1)
            d1 = d1 + colorChannel(data2[:,slices[1],:],1)
            d2 = d2 + colorChannel(data2[:,:,slices[2]],1)
    fig, (p0,p1,p2) = plt.subplots(1,3)
    fig.suptitle(title)
    fig.set_size_inches(16, 7)
    p0.set_title('sagittal')
    p1.set_title('coronal')
    p2.set_title('axial')
    p0.imshow(np.flip(np.transpose(np.array(d0,dtype=np.float32),[1,0,2][:len(d0.shape)]),0))
    p1.imshow(np.flip(np.transpose(np.array(d1,dtype=np.float32),[1,0,2][:len(d1.shape)]),0))
    p2.imshow(np.flip(np.transpose(np.array(d2,dtype=np.float32),[1,0,2][:len(d2.shape)]),0))
    plt.show('block')
    plt.close()

def showRadiomicsDist(title, hist1, hist2, better=False):
    fig, (p0,p1) = plt.subplots(1,2)
    fig.suptitle(title)
    p0.set_title('original')
    p1.set_title('log10')
    fig.set_size_inches(16, 7)
    p0.stairs(hist1[0],hist1[1],fill=True,color='blue')
    p1.stairs(hist2[0],hist2[1],fill=True,color='red' if better else 'blue')
    plt.show('block')
    plt.close()