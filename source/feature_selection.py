#=====================================================================================================================================
import os
# while 'source' not in os.listdir():
#     os.chdir('..')
# os.chdir('source')
FORCE = False
#=====================================================================================================================================
props={
    'path'          : 'data',
    'seed'          : 42,
    'split'         : 0.8,
    'test_split'    : 0.5,
    'control'       : True,
    'huntington'    : False,
    'left'          : False,
    'right'         : False,
    'threshold'     : 0.6,
    'binarize'      : True,
    'not_connected' : True,
    'single'        : None,
    'features'      : [],
    'features_vox'  : [],
    'radiomics'     : [],
    'space'         : 'native',
    'radiomics_vox' : [
        {'im':'t1','fe':['k5_b25','k7_b25','k9_b25','k11_b25','k13_b25','k15_b25','k17_b25','k19_b25','k21_b25']},
    ],
    'rad_vox_norm'  : 'norm',
    'outp'          : 'connectivity',
    'balance_data'  : True,
    'targets_all'   : False,
    'collapse_max'  : False,
    'debug'         : False,
}
architecture={
    'activation'    : 'sigmoid',
    'layers'        : [2048,1024,512,256,128],
    'loss'          : 'CCE',
    'learning_rate' : 0.001,
    'batch_size'    : 100000,
    'patience'      : 7,
}
#=====================================================================================================================================
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import mixed_precision

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])
mixed_precision.set_global_policy('mixed_float16')
#=====================================================================================================================================
import gc
from util import getHashId, pickleSave, pickleLoad, getAccuarcy, predictInBatches
from ModelClassificationFNN import *
from tensorflow.keras.optimizers import Adam

path = props['path']+'/models'

def runModel(props):
    #get data
    gen = DataGenerator(**props)
    train, val, test = gen.getData()
    #get model id and hash
    HASHID, HASH = getHashId(architecture,props)
    #compile model
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=architecture['patience'],
    )
    save = tf.keras.callbacks.ModelCheckpoint(
        filepath=path+'/{}.weights.h5'.format(HASHID),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
    )
    model = buildModel(train[0].shape[1], train[1].shape[1], activation=architecture['activation'], layers=architecture['layers'])
    model.compile(loss=CCE, optimizer=Adam(learning_rate=architecture['learning_rate']), jit_compile=True, metrics=[STD,MAE])
    #train model
    if FORCE or not os.path.exists(path+'/{}.pkl'.format(HASHID)):
        wrapper1 = DataWrapper(train,architecture['batch_size'])
        wrapper2 = DataWrapper(val,architecture['batch_size'],False)
        history = model.fit(wrapper1,
            validation_data=wrapper2,
            epochs=10000,
            verbose=0,
            callbacks = [save,stop],
        )
        pickleSave(path+'/{}.pkl'.format(HASHID), history.history)
        del history
    model.load_weights(path+'/{}.weights.h5'.format(HASHID))
    #return accuracy
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
    del save
    del stop
    del model
    del HASH
    del HASHID
    gc.collect()
    return ac
#=====================================================================================================================================
import numpy as np
from DataGeneratorClassificationFNN import DataGenerator

features_oc = np.load(props['path']+'/preprocessed/features_vox.npy')
features_maxlen = max([len(f) for f in features_oc])
def log(msg):
    msg = str(msg)
    print(msg)
    with open('feature_selection.log','a') as log:
        log.write(msg+'\n')
def logStatus(ite, fea, ac):
    ret = str(ite)
    while len(ret) < 4:
        ret += ' '
    ret += fea
    while len(ret) < features_maxlen:
        ret += ' '
    ret += ' '+str(round(ac*100,1))
    log(ret)

#==== LOAD SAVED ====#
if os.path.exists('state.pkl'):
    state = pickleLoad('state.pkl')
    print(state)
    j0 = state['j']
    i0 = state['i']
    accuracies = state['accuracies']
    excludeds = state['excludeds']
    last_iter_best_idx = state['last_iter_best_idx']
    last_iter_best = state['last_iter_best']
    best_idxs = state['best_idxs']
    features_ex = state['features_ex']
    current_best_idx = state['current_best_idx']
    current_best = state['current_best']
    resumed = True
else:
    j0 = 0
    i0 = 0
    accuracies = []
    excludeds = []
    last_iter_best_idx = 0
    last_iter_best = 0
    best_idxs = []
    features_ex = []
    current_best_idx = -999
    current_best = -999
    resumed = False
    #get baseline of all features
    baseline = runModel(props)
    accuracies.append(baseline)
    excludeds.append([])
    last_iter_best_idx = 0
    last_iter_best = accuracies[0]
    best_idxs.append(0)
    #stuff
    open('feature_selection.log','w').close()
    log('baseline: '+str(round(baseline*100,1)))
#====================#

#top-down exhaustive search
max_iter = len(features_oc)
for j in range(j0,max_iter):
    current_features = [f for f in features_oc if f not in features_ex]
    if resumed:
        resumed = False
    else:
        current_best_idx = -1
        current_best = 0
    for i in range(i0,len(current_features)):
        #==== SAVE ====#
        state = {
            'j':j,
            'i':i,
            'accuracies':accuracies,
            'excludeds':excludeds,
            'last_iter_best_idx':last_iter_best_idx,
            'last_iter_best':last_iter_best,
            'best_idxs':best_idxs,
            'features_ex':features_ex,
            'current_best_idx':current_best_idx,
            'current_best':current_best,
        }
        pickleSave('state.pkl',state)
        del state
        #==============#
        currently_excluded = current_features[i]
        props['features_vox'] = [f for f in current_features if f != currently_excluded]
        ac = runModel(props)
        if ac > current_best:
            current_best = ac
            current_best_idx = len(accuracies)
        accuracies.append(ac)
        excludeds.append(features_ex+[currently_excluded])
        logStatus(i,currently_excluded,ac)
    if current_best < last_iter_best:
        log('Validation accuracy not increasing, stopping!')
        break
    log('===================================')
    last_iter_best_idx = current_best_idx
    last_iter_best = current_best
    best_idxs.append(current_best_idx)
    features_ex = excludeds[current_best_idx]
    log(features_ex)
    logStatus(j,features_ex[-1],accuracies[current_best_idx])
    log('===================================')
#=====================================================================================================================================