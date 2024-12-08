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
    #mixed_precision.set_global_policy('mixed_float16')
    tf.keras.backend.set_floatx('float16')
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

def runModel(train, val, reset_only, hashid, path):
    global model
    global untrained
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=architecture['patience'],
    )
    save = tf.keras.callbacks.ModelCheckpoint(
        filepath=props['path']+'/models/{}.weights.h5'.format(hashid),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
    )
    if reset_only:
        model.set_weights(untrained)
    else:
        model = buildModel(train[0].shape[1], train[1].shape[1], activation=architecture['activation'], layers=architecture['layers'])
        model.compile(loss=CCE, optimizer=Adam(learning_rate=architecture['learning_rate']), jit_compile=True, metrics=[STD,MAE])
        untrained = model.get_weights()
    if FORCE or not os.path.exists(path+'/{}.pkl'.format(hashid)):
        wrapper1 = DataWrapper(train,architecture['batch_size'])
        wrapper2 = DataWrapper(val,architecture['batch_size'],False)
        history = model.fit(wrapper1,
            validation_data=wrapper2,
            epochs=10000,
            verbose=1,
            callbacks = [save,stop],
        )
        model.load_weights(path+'/{}.weights.h5'.format(hashid))
        shutil.copyfile(props['path']+'/models/{}.weights.h5'.format(hashid), path+'/{}.weights.h5'.format(hashid))
        del wrapper1
        del wrapper2
        pickleSave(path+'/{}.pkl'.format(hashid), history.history)
        del history
        del save
        del stop
    else:
        model.load_weights(path+'/{}.weights.h5'.format(hashid))
    ac = getAccuarcy(val[1],predictInBatches(model,val[0],architecture['batch_size']))
    gc.collect()
    return ac

URL = 'https://thesis.nlevi.dev'
TOKEN = '[TOKEN]'
PATH = 'data/models'

def getTask(URL):
    while True:
        response = requests.get(URL+'/task_pop',headers={'Authorization':TOKEN})
        if response.status_code == 200:
            return response.json()
        time.sleep(5)

def postResult(URL,task,ac):
    requests.post(URL+'/task_result',json={'task':task,'result':ac},headers={'Authorization':TOKEN})

def start(URL=URL, PATH=PATH):
    gen = DataGenerator(**props)
    train, val = gen.getData(cnt=2)
    del gen
    last_exc_len = -1
    while True:
        task = getTask(URL)
        print(task)
        feature_mask = np.array([f not in task['excluded'] for f in features_oc], np.bool_)
        feature_mask = np.repeat(feature_mask,train[0].shape[-1]//len(feature_mask))
        train_sliced = train
        train_sliced[0] = train_sliced[0][:,feature_mask]
        val_sliced = val
        val_sliced[0] = val_sliced[0][:,feature_mask]
        ac = runModel(train_sliced,val_sliced,last_exc_len==len(task['excluded']),task['hashid'],PATH)
        print(ac)
        postResult(URL,task,ac)
        last_exc_len = len(task['excluded'])

if __name__ == "__main__":
    start()