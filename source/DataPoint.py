import os
import time
import psutil
import numpy as np
import nibabel as nib
from util import *
from visual import *
import LayeredArray as la

class DataPoint:
    def __init__(self, name, path='data', debug=True, out='console'):
        self.name = name
        self.path = path
        self.debug = debug
        self.ram = -1
        self.tim = time.time()
        self.out = out
        if not os.path.isdir(path+'/preprocessed/'+name):
            self.log('Creating output directory at \'{}\'!'.format(path+'/preprocessed/'+name))
            os.makedirs(path+'/preprocessed/'+name,exist_ok=True)

    def log(self, msg):
        if not self.debug: return
        thr = formatThreadName()
        while len(thr) < 4:
            thr = thr+' '
        ram = round(psutil.Process().memory_info().rss/1024**3,1)
        r = '{}GB'.format(ram)
        while len(r) < 7:
            r = ' '+r
        tim = time.time()
        t = formatTime(tim-self.tim)
        while len(t) < 6:
            t = ' '+t
        o = '{} [DATAPOINT {} {} {}] {}'.format(thr,self.name,r,t,msg)
        if self.out == 'console':
            print(o)
        else:
            with open(self.out,'a') as log:
                log.write(o+'\n')
        self.ram = ram
        self.tim = tim

    def preprocess(self):
        self.tim = time.time()
        self.log('Started preprocessing!')
        #dMRI datadiffusion      
        self.log('Loading diffusion!')
        diffusion      = nib.load(self.path+'/raw/'+self.name+'/diffusion.nii.gz')
        mat_diff       = diffusion.get_sform()
        diffusion      = diffusion.get_fdata()[:,:,:,0]
        #brain mask for dMRI
        self.log('Loading diffusion_mask!')
        diffusion_mask = nib.load(self.path+'/raw/'+self.name+'/diffusion_mask.nii.gz').get_fdata()
        #T1 MRI data
        self.log('Loading t1!')
        t1             = nib.load(self.path+'/raw/'+self.name+'/t1.nii.gz')
        mat_t1         = t1.get_sform()
        t1             = t1.get_fdata()
        #brain mask for T1 MRI
        self.log('Loading t_mask!')
        t1_mask        = nib.load(self.path+'/raw/'+self.name+'/t1_mask.nii.gz').get_fdata()
        t1             = t1 * t1_mask
        #register t1
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            tmp0, space = toSpace(convertToMask(diffusion), mat_diff, None , order=0)
            tmp1, _     = toSpace(convertToMask(t1)       , mat_t1  , space, order=0)
            tmp2 = np.min(np.array([d.shape for d in [tmp0,tmp1]]),0)
            tmp0 = tmp0[0:tmp2[0],0:tmp2[1],0:tmp2[2]]
            tmp1 = tmp1[0:tmp2[0],0:tmp2[1],0:tmp2[2]]
            showSlices(tmp0, tmp1, 'before registration')
            self.tim = self.tim+(time.time()-tmp3)
            del tmp0
            del tmp1
        self.log('Registering t1!')
        mat_t1 = register(diffusion,t1,mat_diff,mat_t1)
        #affine transform
        self.log('Applying affine transformation to diffusion!')
        diffusion     , space = toSpace(diffusion     , mat_diff, None , order=1)
        self.log('Applying affine transformation to t1!')
        t1            , _     = toSpace(t1            , mat_t1  , space, order=1)
        self.log('Applying affine transformation to diffusion_mask!')
        diffusion_mask, _     = toSpace(diffusion_mask, mat_diff, space, order=0)
        self.log('Applying affine transformation to t1_mask!')
        t1_mask       , _     = toSpace(t1_mask       , mat_t1  , space, order=0)
        self.log('Calculating cropped size!')
        shape          = np.min(np.array([d.shape for d in [diffusion,t1]]),0)
        bg_di = convertToMask(diffusion[0:shape[0],0:shape[1],0:shape[2]])
        bg_t1 = t1_mask[0:shape[0],0:shape[1],0:shape[2]]
        bounds = findMaskBounds(np.logical_or(bg_di,bg_t1))
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            bg_di = bg_di[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
            bg_t1 = bg_t1[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
            showSlices(bg_di, bg_t1, 'after registration')
            self.tim = self.tim+(time.time()-tmp3)
        else:
            del bg_di
            del bg_t1
        #========================   diffusion    =======================#
        self.log('Saving diffusion!')
        diffusion = np.array(diffusion[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        np.save(self.path+'/preprocessed/'+self.name+'/diffusion',diffusion)
        del diffusion
        #======================== diffusion_mask =======================#
        self.log('Saving diffusion_mask!')
        diffusion_mask = np.array(diffusion_mask[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        np.save(self.path+'/preprocessed/'+self.name+'/diffusion_mask',diffusion_mask)
        del diffusion_mask
        #========================       t1       =======================#
        self.log('Saving t1!')
        t1 = np.array(t1[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            showSlices(t1)
            self.tim = self.tim+(time.time()-tmp3)
        np.save(self.path+'/preprocessed/'+self.name+'/t1',t1)
        del t1
        #========================    t1_mask     =======================#
        self.log('Saving t1_mask!')
        t1_mask = np.array(t1_mask[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        np.save(self.path+'/preprocessed/'+self.name+'/t1_mask',t1_mask)
        del t1_mask
        #========================      roi       =======================#
        self.log('Loading roi!')
        roi = np.concatenate((
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Basal_G_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Basal_G_Right.nii.gz').get_fdata(),-1),
              ),3)
        self.log('Applying affine transformation to roi!')
        roi = toSpace(roi, mat_diff, space, order=0)[0]
        self.log('Saving roi!')
        roi = np.array(roi[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            showSlices(bg_t1, roi)
            self.tim = self.tim+(time.time()-tmp3)
        la.save(self.path+'/preprocessed/'+self.name+'/roi',roi)
        del roi
        #========================    targets     =======================#
        self.log('Loading cortical targets!')
        tar = np.concatenate((
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Limbic_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Executive_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Rostral_Motor_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Caudal_Motor_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Parietal_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Occipital_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Temporal_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Limbic_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Executive_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Rostral_Motor_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Caudal_Motor_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Parietal_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Occipital_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/Temporal_Right_diff.nii.gz').get_fdata(),-1),
              ),3)
        self.log('Applying affine transformation to cortical targets!')
        tar = toSpace(tar, mat_diff, space, order=0)[0]
        self.log('Saving cortical targets!')
        tar = np.array(tar[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            showSlices(bg_t1, tar)
            self.tim = self.tim+(time.time()-tmp3)
        la.save(self.path+'/preprocessed/'+self.name+'/targets',tar)
        del tar
        #========================  connectivity  =======================#
        self.log('Loading connectivity maps!')
        con = np.concatenate((
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/L_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/E_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/RM_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/CM_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/P_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/O_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/T_relative_connectivity_Left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/L_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/E_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/RM_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/CM_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/P_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/O_relative_connectivity_Right.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/T_relative_connectivity_Right.nii.gz').get_fdata(),-1),
              ),3)
        self.log('Applying affine transformation to connectivity maps!')
        con = toSpace(con, mat_diff, space, order=0)[0]
        self.log('Saving connectivity maps!')
        con = np.array(con[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if self.debug and self.out == 'console':
            tmp3 = time.time()
            showSlices(bg_t1, con)
            self.tim = self.tim+(time.time()-tmp3)
        la.save(self.path+'/preprocessed/'+self.name+'/connectivity',con)
        del con
        self.log('Done preprocessing!')
        #return shape
        return bounds[:,1]-bounds[:,0]

    def radiomicsVoxel(self, kernelWidth=5, binWidth=25, excludeSlow=False, recompute=False):
        self.tim = time.time()
        self.log('Started computing voxel based radiomics!')
        name = 't1_radiomics_raw_k{}_b{}'.format(kernelWidth,binWidth)
        if (not recompute) and (os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'.npy')):
            self.log('Already computed! Skipping!')
            return
        feature_classes = ['firstorder','glcm','glszm','glrlm','ngtdm','gldm']
        t1 = np.load(self.path+'/preprocessed/'+self.name+'/t1.npy')
        t1_mask = np.load(self.path+'/preprocessed/'+self.name+'/t1_mask.npy')
        for feature_class in feature_classes:
            if excludeSlow and feature_class not in ['ngtdm','gldm']: continue
            if (recompute) or (not os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')):
                self.log('Started computing feature class {}!'.format(feature_class))
                r = computeRadiomics(t1, t1_mask, feature_class, voxelBased=True, kernelWidth=kernelWidth, binWidth=binWidth)
                np.save(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class, r)
                del r
                self.log('Done computing feature class {}!'.format(feature_class))
            else:
                self.log('Already computed feature class {}!'.format(feature_class))
        self.log('Saving voxel based radiomics!')
        raw = []
        for feature_class in feature_classes:
            raw.append(np.load(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy'))
        raw = np.concatenate(raw, axis=-1)
        np.save(self.path+'/preprocessed/'+self.name+'/'+name, raw)
        self.log('Deleting partial data!')
        for feature_class in feature_classes:
            if os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy'):
                os.remove(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')
        self.log('Done computing voxel based radiomics!')
    
    def radiomics(self, binWidth=25):
        self.tim = time.time()
        feature_classes = ['firstorder','glcm','glszm','glrlm','ngtdm','gldm','shape']
        t1 = np.load(self.path+'/preprocessed/'+self.name+'/t1.npy')
        for i in range(3):
            if   i == 0:
                self.log('Started computing radiomic features for t1 brain mask!')
                masks = np.expand_dims(np.load(self.path+'/preprocessed/'+self.name+'/t1_mask.npy'),-1)
            elif i == 1:
                self.log('Started computing radiomic features for roi!')
                masks = la.load(self.path+'/preprocessed/'+self.name+'/roi.pkl')
            elif i == 2:
                self.log('Started computing radiomic features for cortical targets!')
                masks = la.load(self.path+'/preprocessed/'+self.name+'/targets.pkl')
            raw1 = []
            for j in range(masks.shape[-1]):
                mask = masks[:,:,:,j]
                raw2 = []
                for feature_class in feature_classes:
                    r = computeRadiomics(t1, mask, feature_class, voxelBased=False, binWidth=binWidth)
                    raw2.append(r)
                raw2 = np.concatenate(raw2, axis=0)
                raw1.append(raw2)
            raw1 = np.array(raw1,np.float32)
            if   i == 0:
                self.log('Done computing radiomic features for t1 brain mask!')
                np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_t1_mask'.format(binWidth),raw1[0])
            elif i == 1:
                self.log('Done computing radiomic features for roi!')
                np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_roi'.format(binWidth),raw1)
            elif i == 2:
                self.log('Done computing radiomic features for cortical targets!')
                np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_targets'.format(binWidth),raw1)
