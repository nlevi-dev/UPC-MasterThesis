#mute warnings
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)

#mute tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#disable unused imports
os.environ['MINIMAL']='2'

#setup available gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
import sys
if __name__ == "__main__" and len(sys.argv) > 1:
    idx = int(sys.argv[1])
    gpus = gpus[idx:idx+1]
    tf.config.set_visible_devices(gpus,'GPU')

#get gpu properties
details = tf.config.experimental.get_device_details(gpus[0])

#create instance identifer
import random
instance=details.get('device_name').replace(' ','')+'_'+str(int(random.random()*10000))
print(instance)

#enable mixed precision for some gpus
print('compute_capability: {}'.format(details.get('compute_capability')[0]))
if not (__name__ == "__main__" and len(sys.argv) > 1):
    if details.get('compute_capability')[0] >= 7:
        from tensorflow.keras import mixed_precision
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

FORCE = True

features_oc = np.load('data/preprocessed/features_vox.npy')

global model
global untrained

global TASK
TASK = {}

ONLINE = os.environ.get('NLEVI_BFG','false')!='true'
SAVE_MODE = os.environ.get('SAVE_MODE','google' if ONLINE else 'local')
URL = 'https://thesis.nlevi.dev' if ONLINE else 'http://127.0.0.1:15000'
TOKEN = '[TOKEN]'
PATH = 'data/models/'

service = None
if SAVE_MODE == 'google':
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    if not os.path.exists('token.json'):
        f = open('token.json','w')
        f.write('[TOKEN]')
        f.close()
    SCOPES = ['https://www.googleapis.com/auth/drive']
    try:
        creds = Credentials.from_authorized_user_file('token.json',SCOPES)
    except:
        creds = False
    if not creds or creds.expired:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('token.json',SCOPES)
            creds = flow.run_console()
        f = open('token.json','w')
        f.write(creds.to_json())
        f.close()
    service = build('drive','v3',credentials=creds)

def googleGetIdOfPath(path):
    if path[0] == '/':  path = path[1:]
    path = path.split('/')
    idxs = ['root']
    for i in range(len(path)):
        results = service.files().list(q="'{}' in parents and trashed=false and name='{}'".format(idxs[i],path[i])).execute()
        idxs.append(results['files'][0]['id'])
    idxs = idxs[1:]
    return idxs[-1]

def googleUpload(file_path, at_directory_path):
    name = file_path.split('/')[-1]
    dir_id = googleGetIdOfPath(at_directory_path)
    mime = 'application/octet-stream'
    meta = {'name':name,'parents':[dir_id],'mimeType':mime}
    media = MediaFileUpload(file_path,mimetype=mime,resumable=True)
    file = service.files().create(body=meta,media_body=media,fields="id").execute()
    return file

def getTask():
    global TASK
    while True:
        try:
            response = requests.get(URL+'/task_pop/'+instance,headers={'Authorization':TOKEN})
            if response.status_code == 200:
                return response.json()
        except:
            pass
        time.sleep(5)

def postResult(ac):
    global TASK
    while True:
        try:
            response = requests.post(URL+'/task_result/'+instance,json={'task':TASK,'result':ac},headers={'Authorization':TOKEN})
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(5)

def keepAlive():
    global TASK
    try:
        requests.post(URL+'/task_keepalive/'+instance,json={'task':TASK},headers={'Authorization':TOKEN})
    except:
        pass

def uploadModel(model_name):
    if SAVE_MODE == 'bfg':
        try:
            requests.post(URL+'/upload/'+model_name+'.weights.h5',files={'file':open(PATH+model_name+'.weights.h5','rb')},headers={'Authorization':TOKEN})
            requests.post(URL+'/upload/'+model_name+'.pkl',files={'file':open(PATH+model_name+'.pkl','rb')},headers={'Authorization':TOKEN})
            os.remove(PATH+model_name+'.weights.h5')
            os.remove(PATH+model_name+'.pkl')
        except:
            print('\nUPLOAD FAILED '+model_name+'!\n')
    elif SAVE_MODE == 'google':
        try:
            googleUpload(PATH+model_name+'.weights.h5','GoogleCluster/MasterThesis/source/data/models')
            googleUpload(PATH+model_name+'.pkl','GoogleCluster/MasterThesis/source/data/models')
            os.remove(PATH+model_name+'.weights.h5')
            os.remove(PATH+model_name+'.pkl')
        except:
            print('\nUPLOAD FAILED '+model_name+'!\n')
    else:
        print('\nUPLOAD FAILED '+model_name+', unknown upload mode!\n')

class KeepAliveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        stat = str(epoch)
        keys = list(logs.keys())
        for k in keys:
            stat += ' - {}: {:.4f}'.format(k,logs[k])
        print(stat, end="\r", flush=True)
        keepAlive()
    
    def on_train_end(self, logs=None):
        print('')

def runModel(train, val, reset_only):
    global TASK

    global model
    global untrained

    if reset_only:
        model.set_weights(untrained)
    else:
        model = buildModel(train[0].shape[1], train[1].shape[1], activation=architecture['activation'], layers=architecture['layers'])
        model.compile(loss=CCE, optimizer=Adam(learning_rate=architecture['learning_rate']), jit_compile=True, metrics=[STD,MAE])
        untrained = model.get_weights()
    if FORCE or not os.path.exists(PATH+TASK['hashid']+'.pkl'):
        wrapper1 = DataWrapper(train,architecture['batch_size'])
        wrapper2 = DataWrapper(val,architecture['batch_size'],False)
        stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=architecture['patience'],
        )
        save = tf.keras.callbacks.ModelCheckpoint(
            filepath=PATH+TASK['hashid']+'.weights.h5',
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
        pickleSave(PATH+TASK['hashid']+'.pkl', history.history)
        model.load_weights(PATH+TASK['hashid']+'.weights.h5')
        if ONLINE and SAVE_MODE != 'local':
            print('Uploading Model!')
            uploadModel(TASK['hashid'])
        del wrapper1.x
        del wrapper2.x
        del wrapper1.y
        del wrapper2.y
        del wrapper1
        del wrapper2
        del history
        del stop
        del save
        del keepalive
    else:
        model.load_weights(PATH+TASK['hashid']+'.weights.h5')
    ac = getAccuarcy(val[1],predictInBatches(model,val[0],architecture['batch_size']))
    gc.collect()
    return ac

def purgeModels():
    histories = os.listdir('data/models')
    histories = [h[:-4] for h in histories if h[-4:] == '.pkl']
    if len(histories) > 0:
        print('Uploading all models!')
        print('0 / '+str(len(histories)), end="\r", flush=True)
    for i in range(len(histories)):
        uploadModel(histories[i])
        print(str(i+1)+' / '+str(len(histories)), end="\r", flush=True)
    if len(histories) > 0:
        print('')

def start():
    if ONLINE and SAVE_MODE != 'local':
        purgeModels()
    global TASK
    last_exc_len = -1
    while True:
        TASK = getTask()
        print(TASK)
        props['features_vox'] = [f for f in features_oc if f not in TASK['excluded']]
        gen = DataGenerator(**props)
        train, val = gen.getData(cnt=2)
        del gen
        ac = runModel(train,val,last_exc_len==len(TASK['excluded']))
        del train
        del val
        gc.collect()
        print(ac)
        postResult(ac)
        last_exc_len = len(TASK['excluded'])

if __name__ == "__main__":
    start()