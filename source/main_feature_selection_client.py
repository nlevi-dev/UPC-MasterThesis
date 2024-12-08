import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
import os
os.environ['MINIMAL']='2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import mixed_precision
gpus = tf.config.experimental.list_physical_devices('GPU')
details = tf.config.experimental.get_device_details(gpus[0])
print('compute_capability: {}'.format(details.get('compute_capability')[0]))
if details.get('compute_capability')[0] >= 7:
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
import gc
import time
import shutil
import requests
from util import pickleSave, getAccuarcy, predictInBatches
from DataGeneratorClassificationFNN import DataGenerator
from ModelClassificationFNN import *
from tensorflow.keras.optimizers import Adam
from main_feature_selection_server import props, architecture

FORCE = False

features_oc = np.load('data/preprocessed/features_vox.npy')

global model
global untrained

global URL
global TOKEN
global PATH
global TASK
URL = 'https://thesis.nlevi.dev'
TOKEN = '[TOKEN]'
PATH = 'data/models'
TASK = {}

def getTask():
    global URL
    global TOKEN
    global PATH
    global TASK
    while True:
        response = requests.get(URL+'/task_pop',headers={'Authorization':TOKEN})
        if response.status_code == 200:
            return response.json()
        time.sleep(5)

def postResult(ac):
    global URL
    global TOKEN
    global PATH
    global TASK
    while True:
        response = requests.post(URL+'/task_result',json={'task':TASK,'result':ac},headers={'Authorization':TOKEN})
        if response.status_code == 200:
            break
        time.sleep(5)

def keepAlive():
    global URL
    global TOKEN
    global PATH
    global TASK
    requests.post(URL+'/task_keepalive',json={'task':TASK},headers={'Authorization':TOKEN})

class KeepAliveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keepAlive()

def runModel(train, val, reset_only):
    global URL
    global TOKEN
    global PATH
    global TASK

    global model
    global untrained

    path_internal = props['path']+'/models/{}'.format(TASK['hashid'])
    path_external = PATH+'/{}'.format(TASK['hashid'])

    if reset_only:
        model.set_weights(untrained)
    else:
        model = buildModel(train[0].shape[1], train[1].shape[1], activation=architecture['activation'], layers=architecture['layers'])
        model.compile(loss=CCE, optimizer=Adam(learning_rate=architecture['learning_rate']), jit_compile=True, metrics=[STD,MAE])
        untrained = model.get_weights()
    if FORCE or not os.path.exists(path_external+'.pkl'):
        wrapper1 = DataWrapper(train,architecture['batch_size'])
        wrapper2 = DataWrapper(val,architecture['batch_size'],False)
        stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=architecture['patience'],
        )
        save = tf.keras.callbacks.ModelCheckpoint(
            filepath=path_internal+'.weights.h5',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
        )
        keepalive = KeepAliveCallback()
        history = model.fit(wrapper1,
            validation_data=wrapper2,
            epochs=10000,
            verbose=0,
            callbacks = [save,stop,keepalive],
        )
        pickleSave(path_external+'.pkl', history.history)
        if path_external != path_internal:
            shutil.copyfile(path_internal+'.weights.h5', path_external+'.weights.h5')
        model.load_weights(path_internal+'.weights.h5')
        del wrapper1
        del wrapper2
        del history
        del save
        del stop
    else:
        model.load_weights(path_external+'.weights.h5')
    ac = getAccuarcy(val[1],predictInBatches(model,val[0],architecture['batch_size']))
    gc.collect()
    return ac

def start(path=PATH):
    global URL
    global TOKEN
    global PATH
    global TASK
    PATH = path
    gen = DataGenerator(**props)
    train, val = gen.getData(cnt=2)
    del gen
    last_exc_len = -1
    while True:
        TASK = getTask()
        print(TASK)
        feature_mask = np.array([f not in TASK['excluded'] for f in features_oc], np.bool_)
        feature_mask = np.repeat(feature_mask,train[0].shape[-1]//len(feature_mask))
        train_sliced = train.copy()
        train_sliced[0] = train_sliced[0][:,feature_mask]
        val_sliced = val.copy()
        val_sliced[0] = val_sliced[0][:,feature_mask]
        ac = runModel(train_sliced,val_sliced,last_exc_len==len(TASK['excluded']))
        print(ac)
        postResult(ac)
        last_exc_len = len(TASK['excluded'])

if __name__ == "__main__":
    start()