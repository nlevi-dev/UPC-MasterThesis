import numpy as np
from DataPoint import DataPoint
from visual import showSlices

DataPoint('C01_1',debug=False,dry_run=True,visualize=True).preprocess()
DataPoint('H33_1',debug=False,dry_run=True,visualize=True).preprocess()

names = np.load('data/preprocessed/names.npy')
for name in names:
    t1 = np.load('data/preprocessed/'+name+'/t1.npy')
    diff = np.load('data/preprocessed/'+name+'/diffusion.npy')
    showSlices(t1,diff,title=name+' preprocessed')