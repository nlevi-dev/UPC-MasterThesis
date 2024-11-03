import os, warnings, math
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from DataGeneratorFFN import reconstruct
from visual import showSlices
import numpy as np

props={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.9,        #train/all ratio
    'test_split'    : 0.0,        #test/(test+validation) ratio
    'control'       : False,      #include control data points
    'huntington'    : True,       #include huntington data points
    'left'          : True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
    'right'         : False,      #include right hemisphere data
    'threshold'     : 0,          #if float value provided, it thresholds the connectivty map
    'binarize'      : True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : False,      #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'single'        : 0,
    'target'        : True,
    'roi'           : True,
    'brain'         : True,
    'features'      : [],
    'features_vox'  : [],
    'radiomics'     : ['b10','b25','b50','b75'],
    'radiomics_vox' : ['k5_b25','k7_b25','k9_b25','k11_b25'],
    'balance_data'  : True,
    'debug'         : False,
}

activation = 'sigmoid'

def buildModel(x_len, y_len, name='FFN'):
    inputs = Input(shape=(x_len,))
    l = Dense(1024, activation=activation)(inputs)
    l = Dense(512, activation=activation)(l)
    l = Dense(128, activation=activation)(l)
    outputs = Dense(y_len, activation='sigmoid' if y_len == 1 else 'softmax')(l)
    model = Model(inputs, outputs, name=name)
    return model

def showResults(model, generator, str = None, threshold=0.5, background=True):
    if str is None:
        showResults(model, generator, 'train', threshold=threshold)
        showResults(model, generator, 'validation', threshold=threshold)
        showResults(model, generator, 'test', threshold=threshold)
        return
    if str == 'train':
        dat = generator.getReconstructor(generator.names[0][0])
    elif str == 'validation':
        dat = generator.getReconstructor(generator.names[1][0])
    elif str == 'test':
        dat = generator.getReconstructor(generator.names[2][0])
    bg = dat[3]
    if not background:
        bg[:,:,:] = 0
    showSlices(bg,reconstruct(dat[1],dat[2],dat[3]),title='{} original ({})'.format(dat[4],str),threshold=threshold)
    predicted = model.predict(dat[0],0,verbose=False)
    showSlices(bg,reconstruct(predicted,dat[2],dat[3]),title='{} predicted ({})'.format(dat[4],str),threshold=threshold)

def MAE(y_true, y_pred):
    error = tf.math.abs(y_true - y_pred)
    return tf.math.reduce_mean(error)

def MSE(y_true, y_pred):
    error = tf.math.square(y_true - y_pred)
    return tf.math.reduce_mean(error)

def MMAE(y_true, y_pred):
    error = tf.math.abs(y_true - y_pred)
    #mask
    error = tf.math.multiply(y_true, error)
    #average
    return tf.math.reduce_mean(error)

def MMSE(y_true, y_pred):
    error = tf.math.abs(y_true - y_pred)
    #mask
    error = tf.math.multiply(y_true, error)
    #square
    error = tf.math.square(error)
    #average
    return tf.math.reduce_mean(error)

def CCE(y_true, y_pred):
    #masked by default
    error = -tf.math.multiply_no_nan(tf.math.log(y_pred), y_true)
    #average
    return tf.math.reduce_mean(tf.math.reduce_sum(error,-1))

def BCE(y_true, y_pred):
    #masked by default
    error = -(tf.math.multiply_no_nan(tf.math.log(y_pred), y_true) + tf.math.multiply_no_nan(tf.math.log(1-y_pred), 1-y_true))
    #average
    return tf.math.reduce_mean(tf.math.reduce_sum(error,-1))

class DataWrapper(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = data[0]
        self.y = data[1]
        self.datalen = len(self.x)
        self.indexes = np.arange(self.datalen)
        self.random = np.random.default_rng(seed)
        self.steps = math.ceil(self.datalen/batch_size)
        if self.shuffle:
            self.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        lo = self.batch_size*idx
        hi = self.batch_size+lo
        if hi > self.datalen:
            hi = self.datalen
        batch_indexes = self.indexes[lo:hi]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            self.random.shuffle(self.indexes)
