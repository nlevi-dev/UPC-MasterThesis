#from RecordHandler import RecordHandler
from visual import showRadiomicsDist
import numpy as np

features_vox = np.load('data/preprocessed/features_vox.npy')
dis = np.load('data/native/preprocessed/t1_features_scale_vox_distributions_k5_b25.npy')
fac = np.load('data/native/preprocessed/t1_features_scale_vox_k5_b25.npy')

for i in range(len(features_vox)):
    showRadiomicsDist(features_vox[i],dis[i][0:2],dis[i][2:4],fac[i][2]=='log10')