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
if details.get('compute_capability')[0] > 7:
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
import gc
import time
import requests
from util import pickleSave, getAccuarcy, predictInBatches
from DataGeneratorClassificationFNN import DataGenerator
from ModelClassificationFNN import *
from tensorflow.keras.optimizers import Adam
from main_feature_selection_server import props, architecture

FORCE = False

features_oc = np.load(props['path']+'/preprocessed/features_vox.npy')

path = props['path']+'/models'

global model

def runModel(props, reset_only, hashid):
    global model
    gen = DataGenerator(**props)
    train, val, test = gen.getData()
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=architecture['patience'],
    )
    save = tf.keras.callbacks.ModelCheckpoint(
        filepath=path+'/{}.weights.h5'.format(hashid),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
    )
    if reset_only:
        model.load_weights('.tmp.weights.h5',skip_mismatch=False)
    else:
        model = buildModel(train[0].shape[1], train[1].shape[1], activation=architecture['activation'], layers=architecture['layers'])
        model.compile(loss=CCE, optimizer=Adam(learning_rate=architecture['learning_rate']), jit_compile=True, metrics=[STD,MAE])
        model.save_weights('.tmp.weights.h5')
    if FORCE or not os.path.exists(path+'/{}.pkl'.format(hashid)):
        wrapper1 = DataWrapper(train,architecture['batch_size'])
        wrapper2 = DataWrapper(val,architecture['batch_size'],False)
        history = model.fit(wrapper1,
            validation_data=wrapper2,
            epochs=10000,
            verbose=0,
            callbacks = [save,stop],
        )
        pickleSave(path+'/{}.pkl'.format(hashid), history.history)
        del history
    model.load_weights(path+'/{}.weights.h5'.format(hashid))
    ac = getAccuarcy(val[1],predictInBatches(model,val[0],architecture['batch_size']))
    del train
    del val
    del test
    del gen
    try:
        wrapper1.x = None
        wrapper1.y = None
        del wrapper1.x
        del wrapper1.y
        wrapper2.x = None
        wrapper2.y = None
        del wrapper2.x
        del wrapper2.y
        del wrapper1
        del wrapper2
    except:
        pass
    gc.collect()
    return ac

URL = 'http://127.0.0.1:15000'

def getTask(URL):
    while True:
        response = requests.get(URL+'/task_pop')
        if response.status_code == 200:
            return response.json()
        time.sleep(5)

def postResult(URL,task,ac):
    requests.post(URL+'/task_result',json={'task':task,'result':ac})

def start(URL=URL):
    last_exc_len = -1
    while True:
        task = getTask(URL)
        print(task)
        if len(task['excluded']) == 0:
            props['features_vox'] = []
        else:
            props['features_vox'] = [f for f in features_oc if f not in task['excluded']]
        ac = runModel(props,last_exc_len==len(task['excluded']),task['hashid'])
        print(ac)
        postResult(URL,task,ac)
        last_exc_len = len(task['excluded'])

if __name__ == "__main__":
    start()