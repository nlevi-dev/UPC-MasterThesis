import os, warnings, math
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Conv3DTranspose, SpatialDropout3D, Concatenate, Multiply
import numpy as np

dropout = 0.3
activation = 'silu'
bias_initializer = 'zeros'
kernel_initializer = 'glorot_uniform'

def doubleConvBlock(x, n_filters):
    x = Conv3D(n_filters, 3, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = SpatialDropout3D(dropout)(x)
    x = Conv3D(n_filters, 3, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = SpatialDropout3D(dropout)(x)
    return x

def downsampleBlock(x, n_filters):
    f = doubleConvBlock(x, n_filters)
    p = MaxPool3D(2)(f)
    return f, p

def upsampleBlock(x, conv_features, n_filters):
    x = Conv3DTranspose(n_filters, 3, 2, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = Concatenate()([x, conv_features])
    x = doubleConvBlock(x, n_filters)
    return x

def buildModel(shape):
    input = Input(shape=shape)
    f1, p1 = downsampleBlock(input, 64)
    f2, p2 = downsampleBlock(p1, 128)
    f3, p3 = downsampleBlock(p2, 256)
    f4, p4 = downsampleBlock(p3, 512)
    bottleneck = doubleConvBlock(p4, 1024)
    u6 = upsampleBlock(bottleneck, f4, 512)
    u7 = upsampleBlock(u6, f3, 256)
    u8 = upsampleBlock(u7, f2, 128)
    u9 = upsampleBlock(u8, f1, 64)
    d1 = Conv3D(64, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(u9)
    d2 = Conv3D(64, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(d1)
    output = Conv3D(1, 1, padding="same", activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(d2)
    mask = Input(shape=shape[:-1]+(1,))
    masked = Multiply()([output, mask])
    model = Model([input,mask], masked, name="unet")
    return model

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

def MAX(_, y_pred):
    return tf.reduce_max(y_pred)

class DataWrapper(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, shuffle=True, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = data[0]
        self.y = data[1]
        self.m = data[3]
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
        m_batch = self.m[batch_indexes]
        y_batch = self.y[batch_indexes]
        return (x_batch,m_batch), y_batch

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            self.random.shuffle(self.indexes)