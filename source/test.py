from DataHandler import DataHandler

handler = DataHandler(
    path='data',
    space='native',
    cores=6,
    out='tmp.log',
    partial=range(0,6),
)
ran = handler.radiomicsVoxel(5, 25, True, True, 't1', fastOnly=True)
if ran: handler.deletePartialData(5, 25, True, 't1')

# handler = DataHandler(path='data', space='normalized', out='console', cores=-1)
# handler.preprocess()
# handler.preloadTarget()

# import nibabel as nib
# import numpy as np
# import scipy.ndimage as ndimage
# data_oc = nib.load('data/MNI152_T1_1mm_brain.nii.gz')
# data2_oc = nib.load('data/MNI152_T1_2mm_brain.nii.gz')
# data = data_oc.get_fdata()
# data = np.where(data > 0, 1, 0)
# data[60:120,60:120,60:120] = 1
# data = ndimage.affine_transform(data,np.linalg.inv(np.dot(np.linalg.inv(data2_oc.get_sform()),data_oc.get_sform())),output_shape=data2_oc.get_fdata().shape,order=0)
# nib.save(nib.MGHImage(data,data2_oc.get_sform(),data2_oc.header),'data/MNI152_T1_2mm_mask.nii.gz')