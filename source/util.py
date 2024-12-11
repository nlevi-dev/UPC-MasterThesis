import os
import sys
import re
import subprocess
import numpy as np
import _pickle as pickle
if int(os.environ.get('MINIMAL','0'))<1:
    from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
    from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
if int(os.environ.get('MINIMAL','0'))<2:
    import SimpleITK as sitk
    from radiomics import featureextractor
    from extractor_params import extractor_params
if int(os.environ.get('MINIMAL','0'))<4:
    import scipy.ndimage as ndimage
    from scipy.stats import pearsonr

def convertToMask(data):
    mask = np.zeros(data.shape,dtype=np.bool_)
    mask = np.where(data == 0, False, True)
    return mask

def getBounds(dim, mat):
    #corners
    cor = np.zeros((8,4))
    cor[:,3] = 1
    cor[1,0:3] = dim[0:3]
    cor[2,0:2] = dim[0:2]
    cor[3,1:3] = dim[1:3]
    cor[4,[0,2]] = dim[[0,2]]
    cor[5,0] = dim[0]
    cor[6,1] = dim[1]
    cor[7,2] = dim[2]
    #calcualte transformed coordinates for every corner
    res = np.array([np.dot(mat,x)[0:3] for x in cor])
    #return min-max values
    return np.append(np.expand_dims(np.min(res,0),0),np.expand_dims(np.max(res,0),0),0)

def toSpace(data, mat, space=None, order=0):
    shape = np.array(data.shape)[0:3]
    #calculate bounds of transformed voxel space (aka world space)
    bounds = getBounds(shape,mat)
    #calculate new shape of world space
    new_shape = np.array(bounds[1]-bounds[0],dtype=np.int32)
    #calculate translation value if not provided
    if space is None:
        space = np.identity(4)
        space[0:3,3] = -1*bounds[0]
    #add translation
    mat = np.dot(space,mat)
    #transfom voxel space
    mat = np.linalg.inv(mat)
    if len(data.shape)==3:
        return (ndimage.affine_transform(data,mat,output_shape=new_shape,order=order), space)
    transformed = np.zeros(tuple(new_shape)+(data.shape[3],),dtype=data.dtype)
    for i in range(data.shape[3]):
        transformed[:,:,:,i] = ndimage.affine_transform(data[:,:,:,i],mat,output_shape=new_shape,order=order)
    return (transformed, space)

def register(data_to, data_from, mat_to, mat_from, stages=[[0,[10,10,5]],[1,[10,10,5]],[2,[1000,1000,100]]]):
    trans = [TranslationTransform3D,RigidTransform3D,AffineTransform3D]
    affreg = AffineRegistration(metric=MutualInformationMetric(32,None),verbosity=0)
    mat = np.identity(4)
    for stage in stages:
        affreg.level_iters = stage[1]
        reg = affreg.optimize(data_to,data_from,trans[stage[0]](),None,mat_to,mat_from,starting_affine=mat)
        mat = reg.affine
    return np.dot(np.linalg.inv(mat),mat_from)

def findMaskBounds(mask, axis=None):
    if axis is None:
        ret = np.zeros((len(mask.shape),2),np.uint16)
        for a in range(ret.shape[0]):
            ret[a,:] = findMaskBounds(mask, a)
        return ret
    mask_zero_columns = np.where(np.sum(mask, axis=axis) == 0, sys.maxsize, 0)
    lower_bound =                    np.min(np.argmax(mask, axis=axis)                     + mask_zero_columns)
    upper_bound = mask.shape[axis] - np.min(np.argmax(np.flip(mask, axis=axis), axis=axis) + mask_zero_columns)
    return np.array([lower_bound, upper_bound],np.uint16)

def computeRadiomicsFeatureLength(feature_classes):
    l = 0
    for feature_class in feature_classes:
        l += len(extractor_params['featureClass'][feature_class])
    return l

def computeRadiomicsFeatureNames(feature_classes):
    f = []
    for feature_class in feature_classes:
        f += [feature_class+'_'+f for f in extractor_params['featureClass'][feature_class]]
    return np.array(f)

def computeRadiomics(data, mask, feature_class, voxelBased=True, kernelWidth=5, binWidth=25, absolute=True):
    params = extractor_params.copy()
    params['voxelSetting']['kernelRadius'] = (kernelWidth-1)//2
    params['setting']['binWidth' if absolute else 'binCount'] = binWidth
    features = params['featureClass'][feature_class]
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(**{feature_class:features})
    sitkData = sitk.GetImageFromArray(np.array(data,np.float32))
    sitkMask = sitk.GetImageFromArray(np.array(mask,np.float32))
    result = extractor.execute(sitkData,sitkMask,voxelBased=voxelBased)
    if voxelBased:
        ret = np.zeros(data.shape+(len(features),),np.float32)
    else:
        ret = np.zeros((len(features),),np.float32)
    for i in range(len(features)):
        r = result['original_{}_{}'.format(feature_class,features[i])]
        if voxelBased:
            o = np.flip(np.array(r.GetOrigin(),np.int32))
            r = np.array(sitk.GetArrayFromImage(r),np.float32)
            ret[o[0]:o[0]+r.shape[0],o[1]:o[1]+r.shape[1],o[2]:o[2]+r.shape[2],i] = r
        else:
            ret[i] = r
    return ret

def getDistribution(data, bins=100, excludeZero=True):
    data = data.flatten()
    if excludeZero:
        data = data[data != 0]
    return np.histogram(data,bins)

def scaleRadiomics(data):
    ret = [0,1,'',0,1]
    scaled1 = data
    mi1 = np.min(scaled1)
    ma1 = np.max(scaled1)
    ret[0] = mi1
    ret[1] = ma1
    scaled1 = (scaled1-mi1)/(ma1-mi1)
    std1 = np.std(scaled1)
    dis1 = getDistribution(scaled1)
    binMax1 = np.max(dis1[0])
    try:
        scaled2 = np.log10(data+1)
        mi2 = np.min(scaled2)
        ma2 = np.max(scaled2)
        scaled2 = (scaled2-mi2)/(ma2-mi2)
        std2 = np.std(scaled2)
        dis2 = getDistribution(scaled2)
        binMax2 = np.max(dis2[0])
    except:
        std2 = 0
        binMax2 = sys.maxsize
        dis2 = [np.zeros(dis1[0].shape,dis1[0].dtype),np.zeros(dis1[1].shape,dis1[1].dtype)]
    if std1 < std2 and binMax1 > binMax2:
        ret[2] = 'log10'
        ret[3] = mi2
        ret[4] = ma2
    return [ret,np.array([np.append(dis1[0],[0]),dis1[1],np.append(dis2[0],[0]),dis2[1]])]

def getHashId(architecture, props, extra=''):
    p = props['path']+'/models/hashes.txt'
    if not os.path.exists(p):
        open(p,'w').close()
    with open(p,'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l != '']
    HASH = getHash([architecture,props])
    if extra != '':
        HASH += '_'+extra
    if HASH not in lines:
        with open(p,'a') as f:
            f.write(HASH+'\n')
        lines.append(HASH)
    HASHID = str(lines.index(HASH))
    while len(HASHID) < 4:
        HASHID = '0'+HASHID
    return [HASHID, HASH]

def getHash(dicts):
    ret = ''
    if isinstance(dicts, dict):
        dicts = list(dicts)
    for j in range(len(dicts)):
        ret += getHashRec(dicts[j])
        if j < len(dicts)-1:
            ret += '_'
    return ret

def getHashRec(data):
    if isinstance(data, dict):
        keys = sorted(list(data.keys()))
        return getHashRec([getHashRec(data[key]) for key in keys])
    elif isinstance(data, list):
        s = ''
        if len(data) == 0:
            return 'e'
        for i in range(len(data)):
            s += getHashRec(data[i])
            if i < len(data)-1:
                s += '_'
        return s
    else:
        if data is None:
            return 'n'
        if isinstance(data, bool):
            return ('1' if data else '0')
        return re.sub('\\W','',str(data))

def pickleLoad(path):
    with open(path,'rb') as f:
        ret = pickle.load(f)
    return ret

def pickleSave(path, obj):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def predictInBatches(model, data, batch_size):
    top = len(data)//batch_size
    arr = [model.predict(data[batch_size*i:batch_size*(i+1)],0,verbose=False) for i in range(top)]
    if len(data) % batch_size != 0:
        arr.append(model.predict(data[batch_size*top:batch_size*(top+1)],0,verbose=False))
    return np.concatenate(arr,0)

def getAccuarcy(y_true, y_pred, mask=None):
    y_true = np.argmax(y_true, -1)
    y_pred = np.argmax(y_pred, -1)
    if mask is None:
        mask = np.ones(y_true.shape)
    else:
        y_true = np.where(mask,y_true,-1)
        y_pred = np.where(mask,y_pred,-2)
    return np.sum(y_true==y_pred)/np.sum(mask)

def getPearson(y_true, y_pred, mask=None):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if mask is not None:
        mask = mask.flatten()
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return pearsonr(y_true, y_pred)[0]

def loadMat(path):
    with open(path,'r') as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'\s+',' ',line)
        line = line.split(' ')
        ret.append(line)
    return np.array(ret, np.float64)

def saveMat(path, data):
    txt = ''
    for line in data:
        for element in line:
            txt += str(element)+'  '
        txt += '\n'
    txt = txt[:-1]
    with open(path,'w') as f:
        f.write(txt)

def applyWarp(input, output, reference, field, extra=''):
    subprocess.call('applywarp -i $(pwd)/{} -o $(pwd)/{} -r $(pwd)/{} -w $(pwd)/{} {}'.format(input,output,reference,field,extra), shell=True)

def maskFromStrings(data, strings):
    if len(data.shape) > 1:
        raise Exception('Data should be 1 dimensional!')
    ret = np.zeros(data.shape,np.bool_)
    for i in range(len(data)):
        if data[i] in strings:
            ret[i] = True
    return ret

def impute(data, fromIdxs, toIdxs=None):
    fromData = data[:,fromIdxs]
    if np.isnan(fromData).any():
        raise Exception('From array must not contain nan(s)!')
    if toIdxs is None:
        toData = data
    else:
        toData = data[:,toIdxs]
    for i in range(len(data)):
        if np.isnan(toData[i,:]).any():
            dists = fromData-np.repeat(fromData[i:i+1,:],len(data),0)
            dists = np.sum(dists**2,1)
            dists[i] = sys.maxsize
            dists = np.argsort(dists)
            closest = 0
            while np.isnan(toData[i,:]).any():
                toData[i,:] = np.where(np.isnan(toData[i,:]),toData[dists[closest],:],toData[i,:])
                closest += 1
    if toIdxs is None:
        return toData
    data[:,toIdxs] = toData
    return data