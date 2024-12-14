import os
import time
import datetime
import psutil
import numpy as np
import scipy.ndimage as ndimage
from util import *
from visual import *
import LayeredArray as la
if int(os.environ.get('MINIMAL','0'))<1:
    import nibabel as nib

missaligned = {
    'tiny'  : ['C11_1','H10_1','H45_1'],
    'large' : ['C32_1','C34_1','C35_1','H33_1','H34_1','H35_1','H36_1','H37_1','H38_1','H39_1','H41_1','H42_1'],
}

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
        self.log('Loading diffusion!')
        diffusion_oc   = nib.load(self.path+'/raw/'+self.name+'/diffusion.nii.gz')
        mat_diff       = diffusion_oc.get_sform()
        diffusion      = diffusion_oc.get_fdata()[:,:,:,-1]
        #T1 MRI data
        self.log('Loading t1!')
        t1_oc          = nib.load(self.path+'/raw/'+self.name+'/t1.nii.gz')
        mat_t1         = t1_oc.get_sform()
        t1             = t1_oc.get_fdata()
        #brain mask for T1 MRI
        self.log('Loading brain_mask!')
        t1_mask        = nib.load(self.path+'/raw/'+self.name+'/t1_mask.nii.gz').get_fdata()
        t1             = t1 * t1_mask
        header_t1 = t1_oc.header
        header_diff = diffusion_oc.header
        header_diff['dim'][0] = 3
        header_diff['dim'][4] = 1
        #save diffusion
        self.log('Saving diffusion!')
        nib.save(nib.MGHImage(diffusion,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/diffusion.nii.gz')
        #dMRI FA, MD and RD data
        exists_diff = os.path.exists(self.path+'/raw/'+self.name+'/diffusion_rd.nii.gz')
        if exists_diff:
            self.log('Loading diffusion_fa!')
            data = nib.load(self.path+'/raw/'+self.name+'/diffusion_fa.nii.gz').get_fdata()
            self.log('Saving diffusion_fa!')
            nib.save(nib.MGHImage(data,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/diffusion_fa.nii.gz')
            self.log('Loading diffusion_md!')
            data = nib.load(self.path+'/raw/'+self.name+'/diffusion_md.nii.gz').get_fdata()
            data = np.where(data < 0, 0, data)
            self.log('Saving diffusion_md!')
            nib.save(nib.MGHImage(data,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/diffusion_md.nii.gz')
            self.log('Loading diffusion_rd!')
            data = nib.load(self.path+'/raw/'+self.name+'/diffusion_rd.nii.gz').get_fdata()
            data = np.where(data < 0, 0, data)
            self.log('Saving diffusion_rd!')
            nib.save(nib.MGHImage(data,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/diffusion_rd.nii.gz')
        #T1/T2 MRI data
        exists_t1t2 = os.path.exists(self.path+'/raw/'+self.name+'/t1t2.nii')
        if exists_t1t2:
            self.log('Loading t1t2!')
            t1t2_oc    = nib.load(self.path+'/raw/'+self.name+'/t1t2.nii')
            mat_t1t2   = t1t2_oc.get_sform()
            t1t2       = t1t2_oc.get_fdata()
            t1t2       = np.where(t1t2 < 0, 0, t1t2)
            t1t2       = np.where(t1t2 > 4, 4, t1t2)
            tmp = t1t2.flatten()
            tmp = tmp[tmp != 0]
            t2t2_upper = np.mean(tmp)+3*np.std(tmp)
            if t2t2_upper < 1: t2t2_upper = 1
            t1t2       = np.where(t1t2 > t2t2_upper, t2t2_upper, t1t2)/t2t2_upper*1000.0
        #register
        if self.name in missaligned['tiny'] or self.name in missaligned['large']:
            self.log('Registring t1!')
            if exists_t1t2:
                if self.name in missaligned['tiny']:
                    mat_t1 = register(t1t2,t1,mat_t1t2,mat_t1,stages=[[1,[1000,1000,100]]])
                else:
                    mat_t1 = register(t1t2,t1,mat_t1t2,mat_t1)
            else:
                if self.name in missaligned['tiny']:
                    mat_t1 = register(diffusion,t1,mat_diff,mat_t1,stages=[[1,[1000,1000,100]]])
                else:
                    mat_t1 = register(diffusion,t1,mat_diff,mat_t1)
        #save t1
        self.log('Saving t1!')
        nib.save(nib.MGHImage(t1,mat_t1,header_t1),self.path+'/native/raw/'+self.name+'/t1.nii.gz')
        #save brain mask
        self.log('Saving brain_mask!')
        nib.save(nib.MGHImage(t1_mask,mat_t1,header_t1),self.path+'/native/raw/'+self.name+'/mask_brain.nii.gz')
        #T1/T2 MRI data
        exists_t1t2 = os.path.exists(self.path+'/raw/'+self.name+'/t1t2.nii')
        if exists_t1t2:
            self.log('Transforming t1t2!')
            t1t2       = ndimage.affine_transform(t1t2,np.linalg.inv(np.dot(np.linalg.inv(mat_t1),mat_t1t2)),output_shape=t1.shape,order=1)
            self.log('Saving t1t2!')
            nib.save(nib.MGHImage(t1t2, mat_t1, header_t1), self.path+'/native/raw/'+self.name+'/t1t2.nii.gz')
        #naming conventions
        names_out = ['limbic','executive','rostral','caudal','parietal','occipital','temporal']
        sides_out = ['left','right']
        names_con = ['L','E','RM','CM','P','O','T']
        names_in = ['Limbic','Executive','Rostral_Motor','Caudal_Motor','Parietal','Occipital','Temporal']
        sides_in = ['Left','Right']
        #save basal masks
        self.log('Processing basal ganglia masks!')
        for i in range(len(sides_in)):
            basal_mask  = nib.load(self.path+'/raw/'+self.name+'/Basal_G_'+sides_in[i]+'.nii.gz').get_fdata()
            nib.save(nib.MGHImage(basal_mask,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/mask_basal_'+sides_out[i]+'.nii.gz')
        #save cortical target masks
        self.log('Processing cortical target masks!')
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                basal_mask  = nib.load(self.path+'/raw/'+self.name+'/'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz').get_fdata()
                nib.save(nib.MGHImage(basal_mask,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/mask_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')
        #save streamline connection
        self.log('Processing streamline connection images!')
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                basal_mask  = nib.load(self.path+'/raw/'+self.name+'/seeds_to_'+names_in[j]+'_'+sides_in[i]+'_diff.nii.gz').get_fdata()
                nib.save(nib.MGHImage(basal_mask,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/streamline_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')
        #save relative connections
        self.log('Processing relative connection images!')
        for i in range(len(sides_out)):
            for j in range(len(names_out)):
                basal_mask  = nib.load(self.path+'/raw/'+self.name+'/'+names_con[j]+'_relative_connectivity_'+sides_in[i]+'.nii.gz').get_fdata()
                nib.save(nib.MGHImage(basal_mask,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/connectivity_'+names_out[j]+'_'+sides_out[i]+'.nii.gz')
        #names = ['caudate','putamen','accumbens']
        idxs = [[11,12,26],[50,51,58]]
        #process basal ganglia segmentation
        exists_basal_seg = os.path.exists(self.path+'/raw/'+self.name+'/basal_seg.nii.gz')
        if exists_basal_seg:
            self.log('Processing basal ganglia segmentation!')
            basal_seg = nib.load(self.path+'/raw/'+self.name+'/basal_seg.nii.gz').get_fdata()
            nib.save(nib.MGHImage(basal_seg,mat_t1,header_t1),self.path+'/native/raw/'+self.name+'/mask_basal_seg.nii.gz')
            labels_oc = None
            for i in range(len(sides_in)):
                basal_mask = nib.load(self.path+'/raw/'+self.name+'/Basal_G_'+sides_in[i]+'.nii.gz').get_fdata()
                if labels_oc is None:
                    mat = np.dot(np.linalg.inv(mat_diff),mat_t1)
                    labels_oc = ndimage.affine_transform(basal_seg,np.linalg.inv(mat),output_shape=basal_mask.shape,order=0)
                labels = np.zeros(labels_oc.shape,np.uint8)
                for j in range(3):
                    labels[labels_oc == idxs[i][j]] = j+1
                for x in range(basal_mask.shape[0]):
                    for y in range(basal_mask.shape[1]):
                        for z in range(basal_mask.shape[2]):
                            if basal_mask[x,y,z] == 0: continue
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
                                if x2 > basal_mask.shape[0]: x2 = basal_mask.shape[0]
                                if y2 > basal_mask.shape[1]: y2 = basal_mask.shape[1]
                                if z2 > basal_mask.shape[2]: z2 = basal_mask.shape[2]
                                m = np.max(labels[x1:x2,y1:y2,z1:z2])
                                if m > 0:
                                    basal_mask[x,y,z] = m
                                    break
                    nib.save(nib.MGHImage(basal_mask,mat_diff,header_diff),self.path+'/native/raw/'+self.name+'/mask_basal_seg_'+sides_out[i]+'.nii.gz')
        #return missing data
        self.log('Done registering!')
        return {
            'name':self.name,
            'diffusion_fa':exists_diff,
            'diffusion_md':exists_diff,
            'diffusion_rd':exists_diff,
            't1t2':exists_t1t2,
            'basal_seg':exists_basal_seg,
        }
    
    def normalize(self):
        self.tim = time.time()
        if (not os.path.exists(self.path+'/raw/'+self.name+'/mat_dif2std.nii.gz')) or (not os.path.exists(self.path+'/raw/'+self.name+'/mat_str2std.nii.gz')):
            return {'name':self.name,'normalized':False}
        mask1 = nib.load('data/MNI152_T1_1mm_mask.nii.gz').get_fdata()
        mask2 = nib.load('data/MNI152_T1_2mm_mask.nii.gz').get_fdata()
        #diffusion trilinear
        diffs = ['diffusion']
        if os.path.exists(self.path+'/native/raw/'+self.name+'/diffusion_fa.nii.gz'):
            diffs += ['diffusion_fa','diffusion_md','diffusion_rd']
        for f in diffs:
            self.log('Normalizing {}!'.format(f))
            out = self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz'
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                out,
                self.path+'/MNI152_T1_2mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_dif2std.nii.gz',
                '--interp=trilinear',
            )
            d = nib.load(out)
            nib.save(nib.MGHImage(d.get_fdata()*mask2,d.get_sform(),d.header),out)
        #diffusion nn
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
            out = self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz'
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                out,
                self.path+'/MNI152_T1_2mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_dif2std.nii.gz',
                '--interp=nn',
            )
            d = nib.load(out)
            nib.save(nib.MGHImage(d.get_fdata()*mask2,d.get_sform(),d.header),out)
        #t1 trilinear
        t1s = ['t1']
        if os.path.exists(self.path+'/native/raw/'+self.name+'/t1t2.nii.gz'):
            t1s += ['t1t2']
        for f in t1s:
            self.log('Normalizing {}!'.format(f))
            out = self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz'
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                out,
                self.path+'/MNI152_T1_1mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_str2std.nii.gz',
                '--interp=trilinear',
            )
            d = nib.load(out)
            nib.save(nib.MGHImage(d.get_fdata()*mask1,d.get_sform(),d.header),out)
        #t1 nn
        t1s = ['mask_brain']
        for f in ['mask_basal_seg']:
            if os.path.exists(self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz'):
                t1s.append(f)
        for f in t1s:
            self.log('Normalizing {}!'.format(f))
            out = self.path+'/normalized/raw/'+self.name+'/'+f+'.nii.gz'
            applyWarp(
                self.path+'/native/raw/'+self.name+'/'+f+'.nii.gz',
                out,
                self.path+'/MNI152_T1_1mm_brain.nii.gz',
                self.path+'/raw/'+self.name+'/mat_str2std.nii.gz',
                '--interp=nn',
            )
            d = nib.load(out)
            nib.save(nib.MGHImage(d.get_fdata()*mask1,d.get_sform(),d.header),out)
        self.log('Done normalizing!')
        return {'name':self.name,'normalized':True}
    
    def inverseWarp(self):
        self.tim = time.time()
        self.log('Started inverse warp field computing!')
        computeInverseWarp(
            self.path+'/raw/'+self.name+'/mat_str2std.nii.gz',
            self.path+'/raw/'+self.name+'/mat_std2str.nii.gz',
            self.path+'/raw/'+self.name+'/t1.nii.gz',
        )
        self.log('Warping normalized coordinate map to native space!')
        applyWarp(
            self.path+'/MNI152_T1_1mm_coords.nii.gz',
            self.path+'/raw/'+self.name+'/coords.nii.gz',
            self.path+'/raw/'+self.name+'/t1.nii.gz',
            self.path+'/raw/'+self.name+'/mat_std2str.nii.gz',
            '--interp=nn',
        )
        self.log('Done!')

    def preprocess(self):
        self.tim = time.time()
        self.log('Started preprocessing!')
        #dMRI data
        self.log('Loading diffusion!')
        diffusion      = nib.load(self.path+'/raw/'+self.name+'/diffusion.nii.gz')
        mat_diff       = diffusion.get_sform()
        diffusion      = diffusion.get_fdata()
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
        if 'normalized' in self.path:
            bounds = np.array([[19,166],[15,206],[1,156]],np.uint8)
        else:
            shape = np.min(np.array([d.shape for d in [diffusion,t1]]),0)
            bg_di = convertToMask(diffusion[0:shape[0],0:shape[1],0:shape[2]])
            bg_t1 = mask_brain[0:shape[0],0:shape[1],0:shape[2]]
            bounds = findMaskBounds(np.logical_or(bg_di,bg_t1))
            del bg_di
            del bg_t1
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
        return bounds

    def radiomicsVoxel(self, feature_class, kernelWidth=5, binWidth=25, recompute=False, absolute=True, inp='t1', data=None, mask=None, cutout=None):
        self.tim = time.time()
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        name = inp+'_radiomics_raw_k{}_b{}'.format(kernelWidth,binstr)
        if data is None:
            data = np.load(self.path+'/preprocessed/'+self.name+'/'+inp+'.npy')
        if mask is None:
            mask = np.load(self.path+'/preprocessed/'+self.name+'/mask_brain.npy')
        if (recompute) or (not os.path.isfile(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class+'.npy')):
            self.log('Started computing voxel based radiomic feature class {}!'.format(feature_class))
            r = computeRadiomics(data, mask, feature_class, voxelBased=True, kernelWidth=kernelWidth, binWidth=binWidth, absolute=absolute)
            if cutout is not None:
                while len(cutout.shape) < 4:
                    cutout = np.expand_dims(cutout,-1)
                concat = []
                for i in range(cutout.shape[-1]):
                    cut = cutout[:,:,:,i].flatten()
                    cutted = np.zeros((np.count_nonzero(cut),r.shape[-1]),np.float32)
                    for j in range(r.shape[-1]):
                        slc = r[:,:,:,j].flatten()
                        cutted[:,j] = slc[cut]
                    concat.append(cutted)
                r = np.concatenate(concat,0)
            if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+name+'_'+feature_class, r)
            self.log('Done computing feature class {}!'.format(feature_class))
        else:
            self.log('Already computed voxel based radiomic feature class {}!'.format(feature_class))
    
    def radiomicsVoxelConcat(self, feature_classes, kernelWidth=5, binWidth=25, absolute=True, inp='t1'):
        self.tim = time.time()
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
        name = inp+'_radiomics_raw_k{}_b{}'.format(kernelWidth,binstr)
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
        binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
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
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}_t1_mask'.format(binstr),raw1[0])
            elif i == 1:
                self.log('Done computing radiomic features for mask_basal!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}_roi'.format(binstr),raw1)
            elif i == 2:
                self.log('Done computing radiomic features for cortical targets!')
                if not self.dry_run: np.save(self.path+'/preprocessed/'+self.name+'/'+inp+'_radiomics_raw_b{}_targets'.format(binstr),raw1)
