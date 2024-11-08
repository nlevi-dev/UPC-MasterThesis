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
    'path'          : 'data',
    'seed'          : 42,
    'split'         : 0.9,
    'test_split'    : 0,
    'control'       : False,
    'huntington'    : True,
    'left'          : True,
    'right'         : False,
    'single'        : 0,
    'target'        : True,
    'roi'           : True,
    'brain'         : True,
    'features'      : [],
    'features_vox'  : [],
    'radiomics'     : ['b10','b75'],
    'radiomics_vox' : ['k5_b25','k11_b25'],
    'debug'         : False,
    'targets_all'   : True,
}

activation = 'silu'

def buildModel(x_len, name='FFN'):
    inputs = Input(shape=(x_len,))
    l = inputs
    for _ in range(10):
        l = Dense(2048, activation=activation)(l)
    outputs = Dense(1, activation=activation)(l)
    model = Model(inputs, outputs, name=name)
    return model

def showResults(model, gen, mode = None, background=True):
    if mode is None:
        showResults(model, gen, 'train')
        showResults(model, gen, 'validation')
        showResults(model, gen, 'test')
        return
    if mode == 'train':
        dat = gen.getReconstructor(gen.names[0][0])
    elif mode == 'validation':
        dat = gen.getReconstructor(gen.names[1][0])
    elif mode == 'test':
        dat = gen.getReconstructor(gen.names[2][0])
    bg = dat[3]
    if not background:
        bg[:,:,:] = 0
    showSlices(bg,reconstruct(dat[1],dat[2],dat[3]),title='{} original ({})'.format(dat[4],mode))
    predicted = model.predict(dat[0],0,verbose=False)
    showSlices(bg,reconstruct(predicted,dat[2],dat[3]),title='{} predicted ({})'.format(dat[4],mode))

def MAE(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    #mask
    error = tf.math.multiply(y_true, error)
    #average
    return tf.math.reduce_mean(error)

def MSE(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    #mask
    error = tf.math.multiply(y_true, error)
    #square
    error = tf.math.square(error)
    #average
    return tf.math.reduce_mean(error)

def STD(_, y_pred):
    return tf.math.reduce_std(y_pred)

def MAX(_, y_pred):
    return tf.math.reduce_max(y_pred)

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