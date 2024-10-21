import time
from DataGenerator import DataGenerator

props={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.8,        #train/all ratio
    'train'         : True,       #training/testing split
    'control'       : True,       #include control data points
    'huntington'    : True,       #include huntington data points
    'batch_size'    : 4,          #batch size
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

def test(props):
    tmp = time.time()
    d = DataGenerator(**props)
    x, y = d.getitem(0)
    print('{}s'.format(int(time.time()-tmp)))
    print(x.shape)
    print(y.shape)

props['spatial'] = False
test(props)

props['left'] = False
props['right'] = True
test(props)

props['left'] = True
props['right'] = False
test(props)

props['left'] = False
props['right'] = False
test(props)
props['left'] = True
props['right'] = True

props['threshold'] = 0.6
props['not_connected'] = False
test(props)

props['not_connected'] = True
test(props)

props['binarize'] = True
props['not_connected'] = False
test(props)

props['not_connected'] = True
test(props)

props['spatial'] = True
test(props)



