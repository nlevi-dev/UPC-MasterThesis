import os, warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from DataGenerator import DataGenerator
from visual import showSlices

props={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.8,        #train/all ratio
    'test_split'    : 0.5,        #test/(test+validation) ratio
    'control'       : True,       #include control data points
    'huntington'    : False,      #include huntington data points
    'type'          : 'CNN',      # FNN CNN FCNN
    'cnn_size'      : 5,          #
    'left'          : True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
    'right'         : False,      #include right hemisphere data
    'threshold'     : 0.5,        #if float value provided, it thresholds the connectivty map
    'binarize'      : True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'single'        : None,       #if int index value is provided, it only returns a specified connectivity map
    'target'        : False,      #
    'roi'           : False,      #
    'brain'         : False,      #
    'features'      : [],         #used radiomics features (emptylist means all)
    'features_vox'  : [],         #used voxel based radiomics features (emptylist means all)
    'radiomics'     : ['b25'],    #used radiomics features bin settings
    'radiomics_vox' : ['k5_b25'], #used voxel based radiomics features kernel and bin settings
    'balance_data'  : True
}
gen = DataGenerator(**props)
train, val, test = gen.getData()
print(train[0][0].shape)
if train[0][1] is not None:
    print(train[0][1].shape)
print(train[1].shape)

# model = Sequential(name='FFN')
# model.add(Input((train[0].shape[1],)))
# model.add(Dense(1024,activation='silu'))
# model.add(Dense(1024,activation='silu'))
# model.add(Dense(1024,activation='silu'))
# model.add(Dense(512,activation='silu'))
# model.add(Dense(512,activation='silu'))
# model.add(Dense(512,activation='silu'))
# model.add(Dense(256,activation='silu'))
# model.add(Dense(256,activation='silu'))
# model.add(Dense(256,activation='silu'))
# model.add(Dense(128,activation='silu'))
# model.add(Dense(128,activation='silu'))
# model.add(Dense(128,activation='silu'))
# model.add(Dense(64,activation='silu'))
# model.add(Dense(64,activation='silu'))
# model.add(Dense(64,activation='silu'))
# model.add(Dense(train[1].shape[1],activation='softmax'))

# callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7)

# model.compile(loss=tf.keras.losses.CategoricalCrossentropy, optimizer='adam', jit_compile=True)
# history = model.fit(train[0], train[1],
#     validation_data=val,
#     batch_size=10000,
#     epochs=100,
#     verbose=1,
#     callbacks = [callback]
# )

# for t in test[0:1]:
#     p = model.predict(t[0])
#     showSlices(t[3],t[2](t[1]),title='original')
#     showSlices(t[3],t[2](p),title='predicted')

