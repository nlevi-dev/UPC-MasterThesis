import os, warnings, math
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np
from DataGeneratorClassificationFNN import reconstruct
if int(os.environ.get('MINIMAL','0'))<2:
    from visual import showSlices

def buildModel(x_len, y_len, name='FFN', activation='sigmoid', layers=[1024,512,128], head_activation=None):
    inputs = Input(shape=(x_len,))
    l = inputs
    for layer in layers:
        l = Dense(layer, activation=activation)(l)
    if head_activation is None:
        a = 'sigmoid' if y_len == 1 else 'softmax'
    else:
        a = head_activation
    outputs = Dense(y_len, activation=a)(l)
    model = Model(inputs, outputs, name=name)
    return model

def showResults(model, generator, mode=None, threshold=0.5, background=True, predict=None):
    if mode is None:
        showResults(model, generator, 'train', threshold, background, predict)
        showResults(model, generator, 'validation', threshold, background, predict)
        showResults(model, generator, 'test', threshold, background, predict)
        return
    idx = {'train':0,'validation':1,'test':2}[mode]
    dat = generator.getReconstructor(generator.names[idx][0])
    if background:
        showSlices(dat[3],reconstruct(dat[1],dat[2],dat[3]),title='{} original ({})'.format(dat[4],mode),threshold=threshold)
    else:
        showSlices(reconstruct(dat[1],dat[2],dat[3]),title='{} original ({})'.format(dat[4],mode),threshold=threshold)
    if predict is None:
        predicted = model.predict(dat[0],0,verbose=False)
    else:
        predicted = predict(mode)
    if background:
        showSlices(dat[3],reconstruct(predicted,dat[2],dat[3]),title='{} predicted ({})'.format(dat[4],mode),threshold=threshold)
    else:
        showSlices(reconstruct(predicted,dat[2],dat[3]),title='{} predicted ({})'.format(dat[4],mode),threshold=threshold)

def STD(_, y_pred):
    return tf.math.reduce_std(y_pred)

def MAE(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    error = tf.math.abs(y_true - y_pred)
    return tf.math.reduce_mean(error)

def MSE(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    error = tf.math.square(y_true - y_pred)
    return tf.math.reduce_mean(error)

def CCE(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    error = -tf.math.multiply_no_nan(tf.math.log(y_pred), y_true)
    return tf.math.reduce_mean(tf.math.reduce_sum(error,-1))

def BCE(y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    error = -(tf.math.multiply_no_nan(tf.math.log(y_pred), y_true) + tf.math.multiply_no_nan(tf.math.log(1-y_pred), 1-y_true))
    return tf.math.reduce_mean(tf.math.reduce_sum(error,-1))

class DataWrapper(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True, seed=42):
        super().__init__()
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