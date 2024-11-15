import os
import time
import datetime
import psutil
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from util import *
from visual import *
import LayeredArray as la

class DataPoint:
    def __init__(self, name, path='data', debug=True, out='console', visualize=False, dry_run=False):
        self.name = name
        self.path = path
        self.debug = debug
        self.ram = -1
        self.tim = time.time()
        self.out = out
        self.visualize = visualize
        self.dry_run = dry_run
        for p in ['/native/raw/','/normalized/raw/','/native/preprocessed/','/normalized/preprocessed/','/native/preloaded/','/normalized/preloaded/']:
            if not os.path.isdir(path+p+name):
                self.log('Creating output directory at \'{}\'!'.format(path+p+name))
                os.makedirs(path+p+name,exist_ok=True)

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
        o = '{}| {} [DATAPOINT {} {} {}] {}'.format(str(datetime.datetime.now())[11:16],thr,self.name,r,t,msg)
        if self.out == 'console':
            print(o)
        else:
            with open(self.out,'a') as log:
                log.write(o+'\n')
        self.ram = ram
        self.tim = tim

    def register(self):
        self.tim = time.time()
        #dMRI data
        diffusion_oc   = nib.load(self.path+'/raw/'+self.name+'/diffusion.nii.gz')
        mat_diff       = diffusion_oc.get_sform()
        diffusion      = diffusion_oc.get_fdata()[:,:,:,0]
        #T1 MRI data
        t1_oc          = nib.load(self.path+'/raw/'+self.name+'/t1.nii.gz')
        mat_t1         = t1_oc.get_sform()
        t1             = t1_oc.get_fdata()
        #brain mask for T1 MRI
        t1_mask_oc     = nib.load(self.path+'/raw/'+self.name+'/t1_mask.nii.gz')
        t1             = t1 * t1_mask_oc.get_fdata()
        #dMRI FA data
        fa_oc          = nib.load(self.path+'/raw/'+self.name+'/diffusion_fa.nii.gz')
        fa             = fa_oc.get_fdata()
        #dMRI MD data
        md_oc          = nib.load(self.path+'/raw/'+self.name+'/diffusion_md.nii.gz')
        md             = md_oc.get_fdata()
        md             = np.where(md < 0, 0, md)
        #dMRI RD data
        rd_oc          = nib.load(self.path+'/raw/'+self.name+'/diffusion_rd.nii.gz')
        mat_rd         = rd_oc.get_sform()
        rd             = rd_oc.get_fdata()
        rd             = np.where(rd < 0, 0, rd)
        #T1/T2 MRI data
        exists_t1t2 = os.path.exists(self.path+'/raw/'+self.name+'/t1t2.nii')
        if not exists_t1t2:
            self.log('t1t2 does not exist!')
        if exists_t1t2:
            t1t2_oc    = nib.load(self.path+'/raw/'+self.name+'/t1t2.nii')
            mat_t1t2   = t1t2_oc.get_sform()
            t1t2       = t1t2_oc.get_fdata()
        #register t1
        self.log('Registering t1!')
        mat_t1 = register(diffusion,t1,mat_diff,mat_t1)
        #register fa
        self.log('Registering rd!')
        mat_rd = register(diffusion,rd,mat_diff,mat_rd)
        #register t1t2
        if exists_t1t2:
            self.log('Registering t1t2!')
            mat_t1t2 = register(diffusion,t1t2,mat_diff,mat_t1t2)
        #save data
        self.log('Saving data!')
        data = nib.MGHImage(diffusion_oc.get_fdata(), mat_diff, diffusion_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/diffusion.nii.gz')
        data = nib.MGHImage(fa, mat_rd, fa_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/diffusion_fa.nii.gz')
        data = nib.MGHImage(md, mat_rd, md_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/diffusion_md.nii.gz')
        data = nib.MGHImage(rd, mat_rd, rd_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/diffusion_rd.nii.gz')
        data = nib.MGHImage(t1, mat_t1, t1_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/t1.nii.gz')
        data = nib.MGHImage(t1_mask_oc.get_fdata(), mat_t1, t1_mask_oc.header)
        nib.save(data, self.path+'/native/raw/'+self.name+'/mask_brain.nii.gz')
        if exists_t1t2:
            data = nib.MGHImage(t1t2, mat_t1t2, t1t2_oc.header)
            nib.save(data, self.path+'/native/raw/'+self.name+'/t1t2.nii.gz')

        names_out = ['limbic','executive','rostral','caudal','parietal','occipital','temporal']
        sides_out = ['left','right']
        names_con = ['L','E','RM','CM','P','O','T']
        names_in = ['Limbic','Executive','Rostral_Motor','Caudal_Motor','Parietal','Occipital','Temporal']
        sides_in = ['Left','Right']

        exists_basal = True
        for i in range(len(sides_in)):
            if not os.path.exists(self.path+'/raw/'+self.name+'/Basal_G_'+sides_in[i]+'.nii.gz'):
                self.log('basal does not exist!')
                exists_basal = False
                break
            raw  = nib.load(self.path+'/raw/'+self.name+'/Basal_G_'+sides_in[i]+'.nii.gz')
            data = nib.MGHImage(raw.get_fdata(), mat_diff, raw.header)
            nib.save(data, self.path+'/native/raw/'+self.name+'/mask_basal_'+sides_out[i]+'.nii.gz')
        
        exists_target = True
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                if not os.path.exists(self.path+'/raw/'+self.name+'/'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz'):
                    self.log('target does not exist!')
                    exists_target = False
                    break
                raw  = nib.load(self.path+'/raw/'+self.name+'/'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz')
                data = nib.MGHImage(raw.get_fdata(), mat_diff, raw.header)
                nib.save(data, self.path+'/native/raw/'+self.name+'/mask_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')
        
        exists_streamline = True
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                if not os.path.exists(self.path+'/raw/'+self.name+'/seeds_to_'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz'):
                    self.log('streamline does not exist!')
                    exists_streamline = False
                    break
                raw  = nib.load(self.path+'/raw/'+self.name+'/seeds_to_'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz')
                data = nib.MGHImage(raw.get_fdata(), mat_diff, raw.header)
                nib.save(data, self.path+'/native/raw/'+self.name+'/streamline_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')
        
        exists_connectivity = True
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                if not os.path.exists(self.path+'/raw/'+self.name+'/'+names_con[j]+'_relative_connectivity_'+sides_in[i]+'.nii.gz'):
                    self.log('connectivity does not exist!')
                    exists_connectivity = False
                    break
                raw  = nib.load(self.path+'/raw/'+self.name+'/'+names_con[j]+'_relative_connectivity_'+sides_in[i]+'.nii.gz')
                data = nib.MGHImage(raw.get_fdata(), mat_diff, raw.header)
                nib.save(data, self.path+'/native/raw/'+self.name+'/connectivity_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')

        names = ['caudate','putamen','accumbens']
        idxs = [[11,12,26],[50,51,58]]

        exists_basal_seg = os.path.exists(self.path+'/raw/'+self.name+'/basal_seg.nii.gz')
        if not exists_basal_seg:
            self.log('basal_seg does not exist!')
        if exists_basal_seg:
            raw  = nib.load(self.path+'/raw/'+self.name+'/basal_seg.nii.gz')
            data = nib.MGHImage(raw.get_fdata(), mat_t1, raw.header)
            nib.save(data, self.path+'/native/raw/'+self.name+'/mask_basal_seg.nii.gz')
            if exists_basal:
                labels_oc = None
                for i in range(len(sides_in)):
                    basal = nib.load(self.path+'/raw/'+self.name+'/Basal_G_'+sides_in[i]+'.nii.gz')
                    data = basal.get_fdata()
                    if labels_oc is None:
                        mat = np.dot(np.linalg.inv(basal.get_sform()),raw.get_sform())
                        labels_oc = ndimage.affine_transform(raw.get_fdata(),np.linalg.inv(mat),output_shape=data.shape,order=0)
                    labels = np.zeros(labels_oc.shape,np.uint8)
                    for j in range(3):
                        labels[labels_oc == idxs[i][j]] = j+1
                    for x in range(data.shape[0]):
                        for y in range(data.shape[1]):
                            for z in range(data.shape[2]):
                                if data[x,y,z] == 0: continue
                                for k in range(100):
                                    m = np.max(labels[x-k:x+k+1,y-k:y+k+1,z-k:z+k+1])
                                    if m > 0:
                                        data[x,y,z] = m
                                        break
                    data = nib.MGHImage(data, mat_diff, basal.header)
                    nib.save(data, self.path+'/native/raw/'+self.name+'/mask_basal_seg_'+sides_out[i]+'.nii.gz')
        
        return {
            'name':self.name,
            'connectivity':exists_connectivity,
            'streamline':exists_streamline,
            't1t2':exists_t1t2,
            'target':exists_target,
            'basal':exists_basal,
            'basal_seg':exists_basal_seg,
        }
    
    def normalize(self):
        pass

    def preprocess(self):
        self.tim = time.time()
        self.log('Started preprocessing!')
        #dMRI data
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
        if self.visualize:
            tmp3 = time.time()
            tmp0, space = toSpace(convertToMask(diffusion), mat_diff, None , order=0)
            tmp1, _     = toSpace(convertToMask(t1)       , mat_t1  , space, order=0)
            tmp2 = np.min(np.array([d.shape for d in [tmp0,tmp1]]),0)
            tmp0 = tmp0[0:tmp2[0],0:tmp2[1],0:tmp2[2]]
            tmp1 = tmp1[0:tmp2[0],0:tmp2[1],0:tmp2[2]]
            showSlices(tmp0, tmp1, title=self.name+' before registration')
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
        if self.visualize:
            tmp3 = time.time()
            bg_di = bg_di[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
            bg_t1 = bg_t1[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]]
            showSlices(bg_di, bg_t1, title=self.name+' after registration')
            self.tim = self.tim+(time.time()-tmp3)
        else:
            del bg_di
            del bg_t1
        #========================   diffusion    =======================#
        self.log('Saving diffusion!')
        diffusion = np.array(diffusion[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/diffusion',diffusion)
        del diffusion
        #======================== diffusion_mask =======================#
        self.log('Saving diffusion_mask!')
        diffusion_mask = np.array(diffusion_mask[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/diffusion_mask',diffusion_mask)
        del diffusion_mask
        #========================       t1       =======================#
        self.log('Saving t1!')
        t1 = np.array(t1[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if self.visualize:
            tmp3 = time.time()
            showSlices(t1, title=self.name+' t1')
            self.tim = self.tim+(time.time()-tmp3)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1',t1)
        del t1
        #========================    t1_mask     =======================#
        self.log('Saving t1_mask!')
        t1_mask = np.array(t1_mask[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1_mask',t1_mask)
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
        if self.visualize:
            tmp3 = time.time()
            showSlices(bg_t1, roi, title=self.name+' roi')
            self.tim = self.tim+(time.time()-tmp3)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/roi',roi)
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
        if self.visualize:
            tmp3 = time.time()
            showSlices(bg_t1, tar, title=self.name+' cortical targets')
            self.tim = self.tim+(time.time()-tmp3)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/targets',tar)
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
        if self.visualize:
            tmp3 = time.time()
            showSlices(bg_t1, con, title=self.name+'connectivity map')
            self.tim = self.tim+(time.time()-tmp3)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/connectivity',con)
        del con
        #=========================  streamline  ========================#
        self.log('Loading streamline maps!')
        sed = np.concatenate((
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Limbic_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Executive_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Rostral_Motor_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Caudal_Motor_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Parietal_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Occipital_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Temporal_Left_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Limbic_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Executive_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Rostral_Motor_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Caudal_Motor_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Parietal_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Occipital_Right_diff.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/seeds_to_Temporal_Right_diff.nii.gz').get_fdata(),-1),
              ),3)
        self.log('Applying affine transformation to streamline maps!')
        sed = toSpace(sed, mat_diff, space, order=0)[0]
        self.log('Saving streamline maps!')
        sed = np.array(sed[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if self.visualize:
            tmp3 = time.time()
            showSlices(bg_t1, sed, title=self.name+'streamline map')
            self.tim = self.tim+(time.time()-tmp3)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/streamline',sed)
        del sed
        self.log('Done preprocessing!')
        return bounds[:,1]-bounds[:,0]

    def radiomicsVoxel(self, feature_class, kernelWidth=5, binWidth=25, recompute=False):
        self.tim = time.time()
        name = 't1_radiomics_raw_k{}_b{}'.format(kernelWidth,binWidth)
        t1 = np.load(self.path+'/preprocessed/'+self.name+'/t1.npy')
        t1_mask = np.load(self.path+'/preprocessed/'+self.name+'/t1_mask.npy')
        if (recompute) or (not os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')):
            self.log('Started computing voxel based radiomic feature class {}!'.format(feature_class))
            r = computeRadiomics(t1, t1_mask, feature_class, voxelBased=True, kernelWidth=kernelWidth, binWidth=binWidth)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class, r)
            self.log('Done computing feature class {}!'.format(feature_class))
        else:
            self.log('Already computed voxel based radiomic feature class {}!'.format(feature_class))
    
    def radiomicsVoxelConcat(self, feature_classes, kernelWidth=5, binWidth=25):
        self.tim = time.time()
        name = 't1_radiomics_raw_k{}_b{}'.format(kernelWidth,binWidth)
        raw = []
        for feature_class in feature_classes:
            r = np.load(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')
            raw.append(r)
        raw = np.concatenate(raw, axis=-1)
        self.log('Saving voxel based radiomics!')
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+name, raw)

    def radiomics(self, binWidth=25):
        # [layer,feature]
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
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_t1_mask'.format(binWidth),raw1[0])
            elif i == 1:
                self.log('Done computing radiomic features for roi!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_roi'.format(binWidth),raw1)
            elif i == 2:
                self.log('Done computing radiomic features for cortical targets!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1_radiomics_raw_b{}_targets'.format(binWidth),raw1)
