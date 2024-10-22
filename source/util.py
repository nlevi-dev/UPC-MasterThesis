import sys
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
from radiomics import featureextractor
from source.extractor_params import extractor_params

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

def register(diffusion, t1, mat_diff, mat_t1):
    affreg = AffineRegistration(metric=MutualInformationMetric(32, None),level_iters=[10,10,5],sigmas=[3.0,1.0,0.0],factors=[4,2,1],verbosity=0)
    translation = affreg.optimize(diffusion,t1,TranslationTransform3D(),None,mat_diff,mat_t1)
    rigid       = affreg.optimize(diffusion,t1,RigidTransform3D()      ,None,mat_diff,mat_t1,starting_affine=translation.affine)
    del translation
    affreg.level_iters = [1000, 1000, 100]
    affine      = affreg.optimize(diffusion,t1,AffineTransform3D()     ,None,mat_diff,mat_t1,starting_affine=rigid.affine)
    del rigid
    return np.dot(np.linalg.inv(affine.affine),mat_t1)

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

def computeRadiomics(data, mask, feature_class, voxelBased=True, kernelWidth=5, binWidth=25):
    params = extractor_params.copy()
    params['voxelSetting']['kernelRadius'] = (kernelWidth-1)//2
    params['setting']['binWidth'] = binWidth
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