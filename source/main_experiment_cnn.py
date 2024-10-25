import os, warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv3D, Reshape, Concatenate
from DataGenerator import DataGenerator, reconstruct
from visual import showSlices

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
}
gen = DataGenerator(**props)
gen.preformatData()
train = gen.getPreformatted('train')
val = gen.getPreformatted('validation')
print(train[0][0].shape)
if len(train[0]) > 1:
    print(train[0][1].shape)
print(train[1].shape)

inputH = Input(shape=train[0][0].shape[1:],name='conv_input')
act = 'relu'
f0 = train[0][0].shape[-1]*2
h = Conv3D(f0,3,strides=1,padding='valid',activation=act,name='conv_1')(inputH)
h = Conv3D(f0,3,strides=1,padding='valid',activation=act,name='conv_2')(h)
h = Reshape((f0,),name='conv_flatten')(h)
if len(train[0]) > 1:
    inputL = Input(shape=train[0][1].shape[1:],name='flat_input')
    f1 = train[0][1].shape[-1]*2
    l = Dense(f1,activation=act,name='flat_1')(inputL)
    l = Dense(f1,activation=act,name='flat_2')(l)
    c = Concatenate(name='head_0')([h,l])
    f = f0+f1
else:
    inputL = None
    c = h
    f = f0
c = Dense(f,activation=act,name='head_1')(c)
c = Dense(f,activation=act,name='head_2')(c)
c = Dense(f,activation=act,name='head_3')(c)
c = Dense(f//2,activation=act,name='head_4')(c)
c = Dense(f//2,activation=act,name='head_5')(c)
c = Dense(f//2,activation=act,name='head_6')(c)
c = Dense(train[1].shape[-1],activation='softmax',name='head_output')(c)

model = Model(inputs=[inputH,inputL][0:len(train[0])], outputs=c)

callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy, optimizer='adam', jit_compile=False)
print('compiled!!!')
history = model.fit(train[0], train[1],
    validation_data=val,
    batch_size=10000,
    epochs=3,
    verbose=1,
    shuffle=False,
    callbacks = [callback]
)


d = gen.getReconstructor(gen.names[0][0])
showSlices(d[3],reconstruct(d[1],d[2],d[3]),title=gen.names[0][0]+' original (train)')
showSlices(d[3],reconstruct(model.predict(d[0]),d[2],d[3]),title=gen.names[0][0]+' predicted (train)')
d = gen.getReconstructor(gen.names[1][0])
showSlices(d[3],reconstruct(d[1],d[2],d[3]),title=gen.names[1][0]+' original (validation)')
showSlices(d[3],reconstruct(model.predict(d[0]),d[2],d[3]),title=gen.names[1][0]+' predicted (validation)')
for n in gen.names[2]:
    d = gen.getReconstructor(n)
    showSlices(d[3],reconstruct(d[1],d[2],d[3]),title=n+' original (test)')
    showSlices(d[3],reconstruct(model.predict(d[0]),d[2],d[3]),title=n+' predicted (test)')