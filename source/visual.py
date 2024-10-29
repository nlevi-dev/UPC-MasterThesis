import re
import os
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
import matplotlib.patches as patches
from matplotlib.path import Path
import xml.etree.ElementTree as ET
from tensorflow.keras.utils import plot_model
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

def showSlices(data1, data2=None, title='', color=False, threshold=0.5):
    slices = [data1.shape[0]//2,data1.shape[1]//2,2*data1.shape[2]//3]
    if data2 is not None and len(data2.shape) == 4:
        colors = pickColors(data2.shape[3])
        if data1.dtype != np.bool_:
            data1 = convertToMask(data1)
        if data2.dtype != np.bool_:
            if threshold == 0:
                bin = np.zeros(data2.shape)
                arged = np.argmax(data2,-1)
                for i in range(bin.shape[-1]):
                    bin[:,:,:,i] = np.where(arged == i, True, False)
                data2 = np.array(bin,dtype=np.float16)
            else:
                data2 = np.array(thresholdArray(data2, threshold),dtype=np.float16)
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
    bins1, edges1 = hist1
    bins2, edges2 = hist2
    if len(bins1) == len(edges1):
        bins1 = bins1[0:len(bins1)-1]
    if len(bins2) == len(edges2):
        bins2 = bins2[0:len(bins2)-1]
    p0.stairs(bins1,edges1,fill=True,color='blue')
    p1.stairs(bins2,edges2,fill=True,color='red' if better else 'blue')
    plt.show('block')
    plt.close()

def plotModel(model):
    try:
        plot_model(model,to_file='tmp.svg',show_shapes=True,show_layer_activations=True)
    except:
        pass
    tree = ET.parse('tmp.svg')
    root = tree.getroot()
    box = root.attrib['viewBox'].split(' ')
    box = [float(v) for v in box]
    transform = root[0].attrib['transform']
    scale = transform[transform.find('scale(')+6:]
    scale = scale[:scale.find(')')]
    scale = scale.split(' ')
    scale = [float(v) for v in scale]
    translate = transform[transform.find('translate(')+10:]
    translate = translate[:translate.find(')')]
    translate = translate.split(' ')
    translate = [float(v) for v in translate]
    width = box[2]-box[0]
    height = box[3]-box[1]
    width_override = float(re.sub('[^0-9]','',root.attrib['width']))
    height_override = float(re.sub('[^0-9]','',root.attrib['height']))
    width_diff = width_override-width
    height_diff = height_override-height
    box[2] += width_diff
    box[1] -= height_diff
    width = box[2]-box[0]
    height = box[3]-box[1]

    def translateCoords(x, y):
        x = float(x)
        y = float(y)
        x *= float(scale[0])
        y *= float(scale[1])
        x += float(translate[0])
        y += float(translate[1])
        x -= float(box[0])
        y -= float(box[1])
        x /= float(width)
        y /= float(height)
        y = 1-y
        return (x, y)

    def processPath(path, verts, codes):
        path = path.strip()
        idx = re.search(r'[ MC]', path)
        if idx is None:
            s = path.split(',')
            verts.append(translateCoords(s[0],s[1]))
            codes.append(Path.LINETO)
            codes[0] = Path.MOVETO
            return
        idx = idx.start()
        if path[idx:idx+1] == 'M' and idx == 0:
            path = path[1:]
            idx = path.index('C')
            s = path[:idx].split(',')
            verts.append(translateCoords(s[0],s[1]))
            codes.append(Path.LINETO)
            path = path[idx+1:]
            idx = path.index(' ')
            s = path[:idx].split(',')
            verts.append(translateCoords(s[0],s[1]))
            codes.append(Path.CURVE3)
            path = path[idx+1:]
            idx = re.search(r'[ MC]', path).start()
            s = path[:idx].split(',')
            verts.append(translateCoords(s[0],s[1]))
            codes.append(Path.CURVE3)
            path = path[idx:]
        else:
            s = path[:idx].split(',')
            verts.append(translateCoords(s[0],s[1]))
            codes.append(Path.LINETO)
            path = path[idx:]
        processPath(path, verts, codes)

    def processNode(node):
        for node in node:
            if node.tag == '{http://www.w3.org/2000/svg}g':
                processNode(node)
            elif node.tag == '{http://www.w3.org/2000/svg}text':
                x = node.attrib['x']
                y = node.attrib['y']
                a = node.attrib['text-anchor']
                if a == 'middle':
                    a = 'center'
                s = float(node.attrib['font-size'])
                f = node.attrib['font-family']
                x, y = translateCoords(x, y)
                p.text(x, y, node.text, horizontalalignment=a, fontsize=s-4)
            elif node.tag in ['{http://www.w3.org/2000/svg}polyline','{http://www.w3.org/2000/svg}polygon','{http://www.w3.org/2000/svg}path']:
                if node.attrib['stroke'] == 'transparent':
                    continue
                points = node.attrib['d' if node.tag == '{http://www.w3.org/2000/svg}path' else 'points']
                fill = node.attrib['fill']
                verts = []
                codes = []
                processPath(points, verts, codes)
                path = Path(verts, codes)
                patch = patches.PathPatch(path, facecolor=fill, lw=2)
                p.add_patch(patch)

    fig, p = plt.subplots(1,1)
    W = 16
    fig.set_size_inches(W, W/width*height)
    p.set_axis_off()
    processNode(root)
    os.remove('tmp.svg')
    plt.show('gamma')
    plt.close()