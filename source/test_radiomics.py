import numpy as np
from visual import showSlices, showRadiomicsDist

names = np.load('data/preprocessed/names.npy')
rad = np.load('data/preprocessed/'+names[0]+'/t1_radiomics_raw_k5_b25.npy')
factors = np.load('data/preprocessed/features_scale_vox_k5_b25.npy')
distributions = np.load('data/preprocessed/features_scale_vox_distributions_k5_b25.npy')
features = np.load('data/preprocessed/features_vox.npy')
for i in range(rad.shape[-1]):
    slice = rad[:,:,:,i]
    showSlices(slice,title=features[i]+' raw (float32)')
    slice1 = slice
    fac = np.array(factors[i][0:2],np.float32)
    slice1 = (slice1-fac[0])/(fac[1]-fac[0])
    showSlices(np.array(slice1,np.float16),title=features[i]+' scaled (float16)')
    slice2 = slice
    if factors[i][2] == 'log10':
        slice2 = np.log10(slice2+1)
        fac = np.array(factors[i][3:5],np.float32)
        slice2 = (slice2-fac[0])/(fac[1]-fac[0])
        showSlices(np.array(slice2,np.float16),title=features[i]+' scaled log10 (float16)')
    showRadiomicsDist(features[i],distributions[i,0:2],distributions[i,2:4],factors[i][2]=='log10')