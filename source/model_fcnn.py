import os, warnings, math
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Conv3DTranspose, concatenate, SpatialDropout3D
from DataGeneratorFCNN import DataGenerator
from visual import showSlices
import numpy as np

props={
    'path'          : 'data',     #path of the data
    'seed'          : 42,         #seed for the split
    'split'         : 0.8,        #train/all ratio
    'test_split'    : 0.5,        #test/(test+validation) ratio
    'control'       : False,      #include control data points
    'huntington'    : True,       #include huntington data points
    'left'          : True,       #include left hemisphere data (if both false, concatenate the left and right hemisphere layers)
    'right'         : False,      #include right hemisphere data
    'threshold'     : None,        #if float value provided, it thresholds the connectivty map
    'binarize'      : True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : False,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'background'    : False,
    'features_vox'  : [],         #used voxel based radiomics features (emptylist means all)
    'radiomics_vox' : ['k5_b25'], #used voxel based radiomics features kernel and bin settings
    'shape'         : (160,208,160),
    'debug'         : False,
}

tmp = props.copy()
tmp['debug'] = True
gen = DataGenerator(**tmp)
train, val, test = gen.getData()

dropout = 0.3
activation = 'softsign'
bias_initializer = 'zeros'
kernel_initializer = 'he_normal'

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
    x = concatenate([x, conv_features])
    x = doubleConvBlock(x, n_filters)
    return x

def buildModel():
    inputs = Input(shape=train[0].shape[1:])
    f1, p1 = downsampleBlock(inputs, 92)
    f2, p2 = downsampleBlock(p1, 128)
    f3, p3 = downsampleBlock(p2, 256)
    f4, p4 = downsampleBlock(p3, 512)
    bottleneck = doubleConvBlock(p4, 1024)
    u6 = upsampleBlock(bottleneck, f4, 512)
    u7 = upsampleBlock(u6, f3, 256)
    u8 = upsampleBlock(u7, f2, 128)
    u9 = upsampleBlock(u8, f1, 64)
    outputs = Conv3D(train[1].shape[-1], 1, padding="same", activation="softmax", kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(u9)
    model = Model(inputs, outputs, name="unet")
    return model

def showResults(model, str = None, threshold=0.5, background=True):
    if str is None:
        showResults(model, 'train', threshold=threshold)
        showResults(model, 'validation', threshold=threshold)
        showResults(model, 'test', threshold=threshold)
        return
    if str == 'train':
        dat = train
    elif str == 'validation':
        dat = val
    elif str == 'test':
        dat = test
    bg = dat[2][0]
    if not background:
        bg[:,:,:] = 0
    showSlices(bg,dat[1][0,:,:,:,:],title='{} original ({})'.format(dat[3][0],str),threshold=threshold)
    predicted = model.predict(dat[0][0:1,:,:,:,:])
    showSlices(bg,predicted[0,:,:,:,:],title='{} predicted ({})'.format(dat[3][0],str),threshold=threshold)

def CustomLoss(weights, shape, batch_size, diversity_weight=1):
    if weights is not None:
        if len(weights) != shape[-1]:
            raise Exception('Mismatched weights and output!')
        s = list(shape)
        s[0] = batch_size
        while len(weights.shape) < len(s):
            r = s[-1*len(weights.shape)-1]
            weights = np.repeat(np.expand_dims(weights,0),r,0)
        weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        error = -tf.math.multiply_no_nan(tf.math.log(y_pred), y_true)
        # error = tf.abs(y_true - y_pred)
        # #mask
        # error = tf.math.multiply(y_true, error)
        # #weight
        # if weights is not None:
        #     error = tf.math.multiply(weights, error)
        # #square
        # error = tf.math.square(error)
        #std error
        std_pred = tf.reshape(y_pred, (-1,y_pred.shape[-1]))
        std_pred = tf.math.reduce_std(std_pred, axis=0)
        summed = tf.reduce_sum(y_true,axis=0)
        summed = tf.reduce_sum(summed,axis=0)
        summed = tf.reduce_sum(summed,axis=0)
        summed = tf.reduce_sum(summed,axis=0)
        mean = summed/tf.reduce_sum(y_true)
        mean = tf.repeat(tf.expand_dims(mean,0),shape[-2],0)
        mean = tf.repeat(tf.expand_dims(mean,0),shape[-3],0)
        mean = tf.repeat(tf.expand_dims(mean,0),shape[-4],0)
        mean = tf.repeat(tf.expand_dims(mean,0),shape[-5],0)
        mask = tf.repeat(tf.reduce_sum(y_true,axis=-1,keepdims=True),shape[-1],-1)
        dev_error = tf.square((y_true-mean)*mask)
        dev_error = tf.reduce_sum(dev_error,axis=0)
        dev_error = tf.reduce_sum(dev_error,axis=0)
        dev_error = tf.reduce_sum(dev_error,axis=0)
        dev_error = tf.reduce_sum(dev_error,axis=0)
        std_true = tf.sqrt(dev_error/tf.reduce_sum(y_true))
        std_error = tf.math.reduce_max(tf.abs(std_pred-std_true))*diversity_weight
        #average
        return tf.math.reduce_sum(error)/tf.math.reduce_sum(y_true)+std_error
    return loss

def MAE(weights, shape, batch_size):
    if weights is not None:
        if len(weights) != shape[-1]:
            raise Exception('Mismatched weights and output!')
        s = list(shape)
        s[0] = batch_size
        while len(weights.shape) < len(s):
            r = s[-1*len(weights.shape)-1]
            weights = np.repeat(np.expand_dims(weights,0),r,0)
        weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        #mask
        error = tf.math.multiply(y_true, error)
        #weight
        if weights is not None:
            error = tf.math.multiply(weights, error)
        #average
        return tf.math.reduce_sum(error)/tf.math.reduce_sum(y_true)
    loss.__name__ = 'MAE'
    return loss

def MSE(weights, shape, batch_size):
    if weights is not None:
        if len(weights) != shape[-1]:
            raise Exception('Mismatched weights and output!')
        s = list(shape)
        s[0] = batch_size
        while len(weights.shape) < len(s):
            r = s[-1*len(weights.shape)-1]
            weights = np.repeat(np.expand_dims(weights,0),r,0)
        weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        error = tf.abs(y_true - y_pred)
        #mask
        error = tf.math.multiply(y_true, error)
        #weight
        if weights is not None:
            error = tf.math.multiply(weights, error)
        #square
        error = tf.math.square(error)
        #average
        return tf.math.reduce_sum(error)/tf.math.reduce_sum(y_true)
    loss.__name__ = 'MSE'
    return loss

def CCE(weights, shape, batch_size):
    if weights is not None:
        if len(weights) != shape[-1]:
            raise Exception('Mismatched weights and output!')
        s = list(shape)
        s[0] = batch_size
        while len(weights.shape) < len(s):
            r = s[-1*len(weights.shape)-1]
            weights = np.repeat(np.expand_dims(weights,0),r,0)
        weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        #already masked by definition
        error = -tf.math.multiply_no_nan(tf.math.log(y_pred), y_true)
        #weight
        if weights is not None:
            error = tf.math.multiply(weights, error)
        #average
        return tf.math.reduce_sum(error)/tf.reduce_sum(y_true)
    loss.__name__ = 'CCE'
    return loss

def MAX(_, y_pred):
    return tf.reduce_max(y_pred)

def STD(_, y_pred):
    flat = tf.reshape(y_pred, (-1,y_pred.shape[-1]))
    std = tf.math.reduce_std(flat, axis=0)
    return tf.reduce_mean(std)

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