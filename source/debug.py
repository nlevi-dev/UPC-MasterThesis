import numpy as np
import tensorflow as tf

true = np.array(np.load('true.npy'),np.float32)
pred = np.array(np.load('pred.npy'),np.float32)

tmp = true
while len(tmp.shape) > 1:
    tmp = np.max(tmp,axis=0)
print(tmp)
tmp = pred
while len(tmp.shape) > 1:
    tmp = np.max(tmp,axis=0)
print(tmp)

weights = None
shape = true.shape

def loss(y_true, y_pred):
    error = tf.abs(y_true - y_pred)
    #mask
    error = tf.math.multiply(y_true, error)
    #weight
    if weights is not None:
        error = tf.math.multiply(weights, error)
    error = tf.math.square(error)
    #average
    return tf.reduce_sum(error)/tf.reduce_sum(y_true)

def STD(y_pred):
    flat = tf.reshape(y_pred, (-1,y_pred.shape[-1]))
    std = tf.math.reduce_std(flat, axis=0)
    return tf.reduce_mean(std)

def STD2(y_true):
    summed = tf.reduce_sum(y_true,axis=0)
    summed = tf.reduce_sum(summed,axis=0)
    #summed = tf.reduce_sum(summed,axis=0)
    summed = tf.reduce_sum(summed,axis=0)
    mean = summed/tf.reduce_sum(y_true)
    mean = tf.repeat(tf.expand_dims(mean,0),y_true.shape[-2],0)
    mean = tf.repeat(tf.expand_dims(mean,0),y_true.shape[-3],0)
    mean = tf.repeat(tf.expand_dims(mean,0),y_true.shape[-4],0)
    #mean = tf.repeat(tf.expand_dims(mean,0),y_true.shape[-5],0)
    mask = tf.repeat(tf.reduce_sum(y_true,axis=-1,keepdims=True),y_true.shape[-1],-1)
    dev_error = tf.square((y_true-mean)*mask)
    dev_error = tf.reduce_sum(dev_error,axis=0)
    dev_error = tf.reduce_sum(dev_error,axis=0)
    dev_error = tf.reduce_sum(dev_error,axis=0)
    #dev_error = tf.reduce_sum(dev_error,axis=0)
    std = tf.sqrt(dev_error/tf.reduce_sum(y_true))
    return tf.reduce_mean(std)

print(STD(pred))
print(STD2(true))