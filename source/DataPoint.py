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
    def __init__(self, name, path='data', debug=True, out='console', visualize=False, dry_run=False, create_folders=False):
        self.name = name
        self.path = path
        self.debug = debug
        self.ram = -1
        self.tim = time.time()
        self.out = out
        self.visualize = visualize
        self.dry_run = dry_run
        if create_folders:
            for p in ['/native/raw/','/normalized/raw/','/native/preprocessed/','/normalized/preprocessed/']:
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
            t1t2       = np.where(t1t2 < 0, 0, t1t2)
            t1t2       = np.where(t1t2 > 1, 1, t1t2)
            mat = np.dot(np.linalg.inv(mat_t1),mat_t1t2)
            print(mat)
            t1t2 = ndimage.affine_transform(t1t2,np.linalg.inv(mat),output_shape=t1.shape,order=0)
        #register t1
        self.log('Registering t1!')
        mat_t1 = register(diffusion,t1,mat_diff,mat_t1)
        #register fa
        self.log('Registering rd!')
        mat_rd = register(diffusion,rd,mat_diff,mat_rd)
        #register t1t2
        # if exists_t1t2:
        #     self.log('Registering t1t2!')
        #     mat_t1t2 = register(diffusion,t1t2,mat_diff,mat_t1t2)
        #alternative way of registering with data loss, but can normalize as a tradeoff
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
                        mat = np.dot(np.linalg.inv(basal.get_sform()),mat_t1)
                        labels_oc = ndimage.affine_transform(raw.get_fdata(),np.linalg.inv(mat),output_shape=data.shape,order=0)
                    labels = np.zeros(labels_oc.shape,np.uint8)
                    for j in range(3):
                        labels[labels_oc == idxs[i][j]] = j+1
                    for x in range(data.shape[0]):
                        for y in range(data.shape[1]):
                            for z in range(data.shape[2]):
                                if data[x,y,z] == 0: continue
                                for k in range(100):
                                    x1 = x-k
                                    x2 = x+k+1
                                    y1 = y-k
                                    y2 = y+k+1
                                    z1 = z-k
                                    z2 = z+k+1
                                    if x1 < 0: x1 = 0
                                    if y1 < 0: y1 = 0
                                    if z1 < 0: z1 = 0
                                    if x2 > data.shape[0]: x2 = data.shape[0]
                                    if y2 > data.shape[1]: y2 = data.shape[1]
                                    if z2 > data.shape[2]: z2 = data.shape[2]
                                    m = np.max(labels[x1:x2,y1:y2,z1:z2])
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
    
    #TODO missing [diffusion_fa, diffusion_md, diffusion_rd, t1t2]
    def normalize(self):
        diffs = ['diffusion']
        for f in diffs:
            self.log('Normalizing {}!'.format(f))
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/MNI152_T1_2mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_dif2std.nii.gz',
                '--interp=trilinear',
            )
        
        diffs = []
        tags = ['mask','connectivity','streamline']
        names = ['limbic','executive','rostral','caudal','parietal','occipital','temporal']
        sides = ['left','right']
        diffs += ['mask_basal_'+s for s in sides]
        for t in tags:
            for s in sides:
                for n in names:
                    f = t+'_'+n+'_'+s
                    if os.path.exists(self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz'):
                        diffs.append(f)
        for f in ['mask_basal_seg_left','mask_basal_seg_right']:
            if os.path.exists(self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz'):
                diffs.append(f)
        for f in diffs:
            self.log('Normalizing {}!'.format(f))
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/MNI152_T1_2mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_dif2std.nii.gz',
                '--interp=nn',
            )
        
        t1s = ['t1']
        for f in t1s:
            self.log('Normalizing {}!'.format(f))
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/MNI152_T1_1mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_str2std.nii.gz',
                '--interp=trilinear',
            )
        
        t1s = ['mask_brain']
        for f in ['mask_basal_seg']:
            if os.path.exists(self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz'):
                t1s.append(f)
        for f in t1s:
            self.log('Normalizing {}!'.format(f))
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz',
                self.path+'/MNI152_T1_1mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_str2std.nii.gz',
                '--interp=nn',
            )

    def preprocess(self, crop_to_bounds=True):
        self.tim = time.time()
        self.log('Started preprocessing!')
        #dMRI data
        self.log('Loading diffusion!')
        diffusion      = nib.load(self.path+'/raw/'+self.name+'/diffusion.nii.gz')
        mat_diff       = diffusion.get_sform()
        diffusion      = diffusion.get_fdata()[:,:,:,0]
        #brain mask for dMRI
        self.log('Loading mask_brain!')
        mask_brain = nib.load(self.path+'/raw/'+self.name+'/mask_brain.nii.gz').get_fdata()
        #T1 MRI data
        self.log('Loading t1!')
        t1             = nib.load(self.path+'/raw/'+self.name+'/t1.nii.gz')
        mat_t1         = t1.get_sform()
        t1             = t1.get_fdata()
        #affine transform
        self.log('Applying affine transformation to diffusion!')
        diffusion     , space = toSpace(diffusion     , mat_diff, None , order=1)
        self.log('Applying affine transformation to t1!')
        t1            , _     = toSpace(t1            , mat_t1  , space, order=1)
        self.log('Applying affine transformation to mask_brain!')
        mask_brain    , _     = toSpace(mask_brain    , mat_t1  , space, order=0)
        self.log('Calculating cropped size!')
        shape          = np.min(np.array([d.shape for d in [diffusion,t1]]),0)
        if crop_to_bounds:
            bg_di = convertToMask(diffusion[0:shape[0],0:shape[1],0:shape[2]])
            bg_t1 = mask_brain[0:shape[0],0:shape[1],0:shape[2]]
            bounds = findMaskBounds(np.logical_or(bg_di,bg_t1))
            del bg_di
            del bg_t1
        else:
            bounds = np.array([[0,shape[0]],[0,shape[1]],[0,shape[2]]])
        #========================   diffusion    =======================#
        self.log('Saving diffusion!')
        diffusion = np.array(diffusion[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        np.save(self.path+'/preprocessed/'+self.name+'/diffusion',diffusion)
        del diffusion
        #========================       t1       =======================#
        self.log('Saving t1!')
        t1 = np.array(t1[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1',t1)
        del t1
        #========================      t1t2      =======================#
        if os.path.exists(self.path+'/raw/'+self.name+'/t1t2.nii.gz'):
            self.log('Loading t1t2!')
            t1t2 = nib.load(self.path+'/raw/'+self.name+'/t1t2.nii.gz')
            self.log('Applying affine transformation to t1t2!')
            t1t2 = toSpace(t1t2.get_fdata(), t1t2.get_sform(), space, order=1)[0]
            self.log('Saving t1t2!')
            t1t2 = np.array(t1t2[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/t1t2',t1t2)
            del t1t2
        #========================       fa       =======================#
        if os.path.exists(self.path+'/raw/'+self.name+'/diffusion_fa.nii.gz'):
            self.log('Loading diffusion_fa!')
            fa = nib.load(self.path+'/raw/'+self.name+'/diffusion_fa.nii.gz')
            self.log('Applying affine transformation to diffusion_fa!')
            fa = toSpace(fa.get_fdata(), fa.get_sform(), space, order=1)[0]
            self.log('Saving diffusion_fa!')
            fa = np.array(fa[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/diffusion_fa',fa)
            del fa
        #========================       md       =======================#
        if os.path.exists(self.path+'/raw/'+self.name+'/diffusion_md.nii.gz'):
            self.log('Loading diffusion_md!')
            md = nib.load(self.path+'/raw/'+self.name+'/diffusion_md.nii.gz')
            self.log('Applying affine transformation to diffusion_md!')
            md = toSpace(md.get_fdata(), md.get_sform(), space, order=1)[0]
            self.log('Saving diffusion_md!')
            md = np.array(md[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/diffusion_md',md)
            del md
        #========================       rd       =======================#
        if os.path.exists(self.path+'/raw/'+self.name+'/diffusion_rd.nii.gz'):
            self.log('Loading diffusion_rd!')
            rd = nib.load(self.path+'/raw/'+self.name+'/diffusion_rd.nii.gz')
            self.log('Applying affine transformation to diffusion_rd!')
            rd = toSpace(rd.get_fdata(), rd.get_sform(), space, order=1)[0]
            self.log('Saving diffusion_rd!')
            rd = np.array(rd[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/diffusion_rd',rd)
            del rd
        #========================    basal_seg   =======================#
        if os.path.exists(self.path+'/raw/'+self.name+'/mask_basal_seg_left.nii.gz'):
            self.log('Loading basal_seg!')
            left = nib.load(self.path+'/raw/'+self.name+'/mask_basal_seg_left.nii.gz')
            right = nib.load(self.path+'/raw/'+self.name+'/mask_basal_seg_right.nii.gz')
            self.log('Applying affine transformation to basal_seg!')
            left = toSpace(left.get_fdata(), left.get_sform(), space, order=0)[0]
            right = toSpace(right.get_fdata(), right.get_sform(), space, order=0)[0]
            self.log('Saving basal_seg!')
            left = np.array(left[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            right = np.array(right[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
            basal_seg = np.zeros(left.shape+(6,),np.bool_)
            for i in range(3):
                basal_seg[:,:,:,i] = (left == (i+1))
                basal_seg[:,:,:,i+3] = (right == (i+1))
            if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/basal_seg',basal_seg)
            del left
            del right
            del basal_seg
        #========================   mask_brain   =======================#
        self.log('Saving mask_brain!')
        mask_brain = np.array(mask_brain[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/mask_brain',mask_brain)
        del mask_brain
        #========================   mask_basal   =======================#
        self.log('Loading mask_basal!')
        mask_basal = np.concatenate([
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/mask_basal_left.nii.gz').get_fdata(),-1),
                np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/mask_basal_right.nii.gz').get_fdata(),-1),
              ],3)
        self.log('Applying affine transformation to mask_basal!')
        mask_basal = toSpace(mask_basal, mat_diff, space, order=0)[0]
        self.log('Saving mask_basal!')
        mask_basal = np.array(mask_basal[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/mask_basal',mask_basal)
        del mask_basal
        #========================    targets     =======================#
        labels = ['limbic','executive','rostral','caudal','parietal','occipital','temporal']
        self.log('Loading cortical targets!')
        tar = np.concatenate(
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/mask_'+f+'_left.nii.gz').get_fdata(),-1) for f in labels]+
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/mask_'+f+'_right.nii.gz').get_fdata(),-1) for f in labels],3)
        self.log('Applying affine transformation to cortical targets!')
        tar = toSpace(tar, mat_diff, space, order=0)[0]
        self.log('Saving cortical targets!')
        tar = np.array(tar[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.bool_)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/targets',tar)
        del tar
        #========================  connectivity  =======================#
        self.log('Loading connectivity maps!')
        con = np.concatenate(
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/connectivity_'+f+'_left.nii.gz').get_fdata(),-1) for f in labels]+
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/connectivity_'+f+'_right.nii.gz').get_fdata(),-1) for f in labels],3)
        self.log('Applying affine transformation to connectivity maps!')
        con = toSpace(con, mat_diff, space, order=0)[0]
        self.log('Saving connectivity maps!')
        con = np.array(con[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/connectivity',con)
        del con
        #=========================  streamline  ========================#
        self.log('Loading streamline maps!')
        sed = np.concatenate(
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/streamline_'+f+'_left.nii.gz').get_fdata(),-1) for f in labels]+
                [np.expand_dims(nib.load(self.path+'/raw/'+self.name+'/streamline_'+f+'_right.nii.gz').get_fdata(),-1) for f in labels],3)
        self.log('Applying affine transformation to streamline maps!')
        sed = toSpace(sed, mat_diff, space, order=0)[0]
        self.log('Saving streamline maps!')
        sed = np.array(sed[bounds[0,0]:bounds[0,1],bounds[1,0]:bounds[1,1],bounds[2,0]:bounds[2,1]],np.float16)
        if not self.dry_run: la.save(self.path+'/preprocessed/'+self.name+'/streamline',sed)
        del sed
        self.log('Done preprocessing!')
        return bounds[:,1]-bounds[:,0]

    def radiomicsVoxel(self, feature_class, kernelWidth=5, binWidth=25, recompute=False, absolute=True, inp='t1'):
        self.tim = time.time()
        name = inp+'_radiomics_raw_k{}_b{}{}'.format(kernelWidth,binWidth,'' if absolute else 'r')
        t1 = np.load(self.path+'/preprocessed/'+self.name+'/'+inp+'.npy')
        t1_mask = np.load(self.path+'/preprocessed/'+self.name+'/mask_brain.npy')
        if (recompute) or (not os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')):
            self.log('Started computing voxel based radiomic feature class {}!'.format(feature_class))
            r = computeRadiomics(t1, t1_mask, feature_class, voxelBased=True, kernelWidth=kernelWidth, binWidth=binWidth, absolute=absolute)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class, r)
            self.log('Done computing feature class {}!'.format(feature_class))
        else:
            self.log('Already computed voxel based radiomic feature class {}!'.format(feature_class))
    
    def radiomicsVoxelConcat(self, feature_classes, kernelWidth=5, binWidth=25, absolute=True, inp='t1'):
        self.tim = time.time()
        name = inp+'_radiomics_raw_k{}_b{}{}'.format(kernelWidth,binWidth,'' if absolute else 'r')
        raw = []
        for feature_class in feature_classes:
            r = np.load(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')
            raw.append(r)
        raw = np.concatenate(raw, axis=-1)
        self.log('Saving voxel based radiomics!')
        if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+name, raw)

    def radiomics(self, binWidth=25, absolute=True, inp='t1'):
        # [layer,feature]
        self.tim = time.time()
        feature_classes = ['firstorder','glcm','glszm','glrlm','ngtdm','gldm','shape']
        t1 = np.load(self.path+'/preprocessed/'+self.name+'/'+inp+'.npy')
        for i in range(3):
            if   i == 0:
                self.log('Started computing radiomic features for mask_brain!')
                masks = np.expand_dims(np.load(self.path+'/preprocessed/'+self.name+'/mask_brain.npy'),-1)
            elif i == 1:
                self.log('Started computing radiomic features for mask_basal!')
                masks = la.load(self.path+'/preprocessed/'+self.name+'/mask_basal.pkl')
            elif i == 2:
                self.log('Started computing radiomic features for cortical targets!')
                masks = la.load(self.path+'/preprocessed/'+self.name+'/targets.pkl')
            raw1 = []
            for j in range(masks.shape[-1]):
                mask = masks[:,:,:,j]
                raw2 = []
                for feature_class in feature_classes:
                    r = computeRadiomics(t1, mask, feature_class, voxelBased=False, binWidth=binWidth, absolute=absolute)
                    raw2.append(r)
                raw2 = np.concatenate(raw2, axis=0)
                raw1.append(raw2)
            raw1 = np.array(raw1,np.float32)
            if   i == 0:
                self.log('Done computing radiomic features for mask_brain!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}{}_t1_mask'.format(binWidth,'' if absolute else 'r'),raw1[0])
            elif i == 1:
                self.log('Done computing radiomic features for mask_basal!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}{}_roi'.format(binWidth,'' if absolute else 'r'),raw1)
            elif i == 2:
                self.log('Done computing radiomic features for cortical targets!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}{}_targets'.format(binWidth,'' if absolute else 'r'),raw1)
