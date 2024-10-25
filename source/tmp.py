from DataGenerator import DataGenerator
props={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.8,        #train/all ratio
    'test_split'    : 0.3,        #test/(test+validation) ratio
    'control'       : True,       #include control data points
    'huntington'    : False,      #include huntington data points
    'type'          : 'CNN',      # FNN CNN FCNN
    'cnn_size'      : 5,          #
    'left'          : True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
    'right'         : False,      #include right hemisphere data
    'threshold'     : 0.6,        #if float value provided, it thresholds the connectivty map
    'binarize'      : True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'single'        : None,       #if int index value is provided, it only returns a specified connectivity map
    'target'        : False,      #
    'roi'           : False,      #
    'brain'         : False,      #
    'features'      : [],         #used radiomics features (emptylist means all)
    'features_vox'  : [],         #used voxel based radiomics features (emptylist means all)
    'radiomics'     : ['b25'],    #used radiomics features bin settings
    'radiomics_vox' : ['k7_b25'], #used voxel based radiomics features kernel and bin settings
    'balance_data'  : True,
    'debug'         : True,
}
gen = DataGenerator(**props)
gen.preformatData()