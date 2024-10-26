import os, warnings, math
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Dropout, Conv3DTranspose, concatenate, Activation
from DataGeneratorFCNN import DataGenerator
from visual import showSlices, plotModel
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
    'threshold'     : 0.6,        #if float value provided, it thresholds the connectivty map
    'binarize'      : True,       #only works if threshold if greater or equal than half, and then it binarizes the connectivity map
    'not_connected' : True,       #only works if thresholded and not single, and then it appends an extra encoding for the 'not connected'
    'background'    : True,
    'features_vox'  : [],         #used voxel based radiomics features (emptylist means all)
    'radiomics_vox' : ['k5_b25'], #used voxel based radiomics features kernel and bin settings
    'shape'         : (160,208,160),
    'debug'         : True,
}
gen = DataGenerator(**props)
train, val, test = gen.getData()

weights = []
for i in range(train[1].shape[-1]-1):
    weights.append(np.count_nonzero(train[1][:,:,:,:,i]))
weights.append(0)
weights = np.array(weights,np.float32)
weights /= np.sum(weights)
weights = weights*-1+1
weights[-1] = 0
weights *= 1000

def double_conv_block(x, n_filters):
    x = Conv3D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = Conv3D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPool3D(2)(f)
    p = Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    x = Conv3DTranspose(n_filters, 3, 2, padding = "same", activation = "relu")(x)
    x = concatenate([x, conv_features])
    x = Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

inputs = Input(shape=train[0].shape[1:])
f1, p1 = downsample_block(inputs, 92)
f2, p2 = downsample_block(p1, 128)
f3, p3 = downsample_block(p2, 256)
f4, p4 = downsample_block(p3, 512)
bottleneck = double_conv_block(p4, 1024)
u6 = upsample_block(bottleneck, f4, 512)
u7 = upsample_block(u6, f3, 256)
u8 = upsample_block(u7, f2, 128)
u9 = upsample_block(u8, f1, 64)
partial_outputs = Conv3D(train[1].shape[-1]-1, 1, padding="same", activation = "softmax")(u9)
bg_inputs = Input(shape=train[0].shape[1:-1]+(1,))*999
bg_outputs = concatenate([partial_outputs, bg_inputs])
outputs = Activation(activation = "softmax")(bg_outputs)
model = Model([inputs, bg_inputs], outputs, name="unet")

def showResults(model, str = None):
    if str is None:
        showResults(model, 'train')
        showResults(model, 'validation')
        showResults(model, 'test')
        return
    if str == 'train':
        dat = train
    elif str == 'validation':
        dat = val
    elif str == 'test':
        dat = test
    showSlices(dat[2][0],dat[1][0,:,:,:,:-1],title='{} original ({})'.format(dat[3][0],str))
    predicted = model.predict([dat[0][0:1,:,:,:,:],dat[1][0:1,:,:,:,-1:]])
    showSlices(dat[2][0],predicted[0,:,:,:,:-1],title='{} predicted ({})'.format(dat[3][0],str))

def weighted_mean_absolute_error(weights, shape, batch_size):
    if len(weights) != shape[-1]:
        raise Exception('Mismatched weights and output!')
    s = list(shape)
    s[0] = batch_size
    while len(weights.shape) < len(s):
        r = s[-1*len(weights.shape)-1]
        weights = np.repeat(np.expand_dims(weights,0),r,0)
    weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        abs = tf.abs(y_true - y_pred)
        weighted = tf.math.multiply(weights, abs)
        return tf.reduce_mean(weighted)
    return loss

def weighted_categorical_cross_entropy(weights, shape, batch_size):
    if len(weights) != shape[-1]:
        raise Exception('Mismatched weights and output!')
    s = list(shape)
    s[0] = batch_size
    while len(weights.shape) < len(s):
        r = s[-1*len(weights.shape)-1]
        weights = np.repeat(np.expand_dims(weights,0),r,0)
    weights = tf.convert_to_tensor(weights)
    def loss(y_true, y_pred):
        log_y_pred = tf.math.log(y_pred)
        element_wise = -tf.math.multiply_no_nan(log_y_pred, y_true)
        weighted = tf.math.multiply(weights, element_wise)
        return tf.reduce_mean(tf.reduce_sum(weighted,axis=-1))
    return loss

class Data(tf.keras.utils.Sequence):
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
        return (x_batch, y_batch[:,:,:,:,-1:]), y_batch
    
    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            self.random.shuffle(self.indexes)

batch_size = 1

loss = weighted_mean_absolute_error(weights,train[1].shape,batch_size)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 7)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FCNN.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model.compile(loss=loss, optimizer='adam', jit_compile=True)

plotModel(model)

# history = model.fit(Data(train,batch_size),
#     validation_data=Data(val,batch_size,False),
#     epochs=100,
#     verbose=1,
#     callbacks = [stop,save]
# )

# showResults(model)