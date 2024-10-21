import os
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
from DataGenerator import DataGenerator

props_og={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.8,        #train/all ratio
    'train'         : True,       #training/testing split
    'control'       : True,       #include control data points
    'huntington'    : True,       #include huntington data points
    'batch_size'    : 1,          #batch size
    'spatial'       : False,      #keep spaital format of flatten voxels in the brain region
    'left'          : True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
    'right'         : True,       #include right hemisphere data
    'threshold'     : False,      #if float value provided, it thresholds the connectivty map
    'binarize'      : False,      #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : False,      #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'single'        : False,      #if int index value is provided, it only returns a specified connectivity map
    'target'        : False,
    'roi'           : False,
    'brain'         : False,
    'features'      : [],         #used radiomics features (emptylist means all)
    'features_vox'  : [],         #used voxel based radiomics features (emptylist means all)
    'radiomics'     : ['b25'],    #used radiomics features bin settings
    'radiomics_vox' : ['k5_b25'], #used voxel based radiomics features kernel and bin settings
}
props=props_og.copy()
max_len = 0
for key in props.keys():
    if len(key) > max_len:
        max_len = len(key)
max_len += 1

def printProps(props1, props2=None):
    if props2 is None:
        print('========================================')
        for key in props1.keys():
            k = key
            while len(k) < max_len:
                k = k+' '
            print('{}: {}'.format(k,props1[key]))
        print('----------------------------------------')
    else:
        p2 = {}
        for key in props1.keys():
            if props1[key] != props2[key]:
                p2[key] = props1[key]
        if len(p2.keys()) == 0:
            printProps(props1)
        else:
            printProps(p2)

shape = np.max(np.load('data/preprocessed/shapes.npy'),0)
features = np.load('data/preprocessed/features.npy')
features_vox = np.load('data/preprocessed/features_vox.npy')
labels = np.load('data/preprocessed/labels.npy')

def test(props):
    printProps(props)
    tmp = time.time()
    d = DataGenerator(dummy=True,**props)
    x, y = d.getitem(0)
    print('{}s'.format(int(time.time()-tmp)))
    print(x.shape)
    print(y.shape)
    if props['spatial']:
        assert props['batch_size'] == x.shape[0]
        assert shape[0] == x.shape[1]
        assert shape[1] == x.shape[2]
        assert shape[2] == x.shape[3]
        if len(props['features_vox']) == 0:
            f = len(features_vox)
        else:
            f = len(props['features_vox'])
        f = f*len(props['radiomics_vox'])
        assert f == x.shape[4]
        assert props['batch_size'] == y.shape[0]
        assert shape[0] == y.shape[1]
        assert shape[1] == y.shape[2]
        assert shape[2] == y.shape[3]
        if props['single'] == False:
            l = len(labels)*2
            if (not props['left']) or (not props['right']):
                l = l//2
            if props['not_connected'] and props['threshold'] != False and props['threshold'] >= 0.5:
                l = l+1
        else:
            l = 1
        assert l == y.shape[4]
    print('========================================\n\n\n')


slave=[
    {'batch_size':1},
    {'batch_size':4},
    {'left':False,
     'right':True},
    {'left':True,
     'right':False},
    {'left':False,
     'right':False},
    {'threshold':0.5,
     'not_connected':True},
    {'single':5},
    lambda props: {'features_vox':features_vox[5:10]} if props['spatial'] else {'features':features[5:10]},
    lambda props: {'radiomics_vox':['k5_b25','k5_b25']} if props['spatial'] else {'radiomics':['b25','b25']},
]

master=[
    {'spatial':[True]},
]

for ma in master:
    props0=props_og.copy()
    for mk in ma.keys():
        for mv in ma[mk]:
            props0[mk] = mv
            for sa in slave:
                props1 = props0.copy()
                if callable(sa) and sa.__name__ == "<lambda>":
                    sa = sa(props1)
                for sk in sa.keys():
                    props1[sk] = sa[sk]
                test(props1)