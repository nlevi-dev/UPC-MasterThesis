import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24576)])

import gc
import numpy as np
from util import getHashId, pickleSave, pickleLoad, getAccuarcy, predictInBatches
from DataGeneratorClassificationFNN import DataGenerator
from ModelClassificationFNN import *
from tensorflow.keras.optimizers import Adam

FORCE = False

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

path = props['path']+'/models'

def runModel(props):
    gen = DataGenerator(**props)
    train, val, test = gen.getData()
    HASHID, HASH = getHashId(architecture,props)
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

STATENAME = 'state1.pkl'

if os.path.exists(STATENAME):
    state = pickleLoad(STATENAME)
    accuracies = state['accuracies']
    excludeds = state['excludeds']
else:
    accuracies = []
    excludeds = []
    baseline = runModel(props)
    accuracies.append([baseline])
    accuracies.append([])
    excludeds.append([[]])
    excludeds.append([])
    pickleSave(STATENAME,{'accuracies':accuracies,'excludeds':excludeds})
    open('feature_selection.log','w').close()
    logStatus(0,'BASELINE',baseline)

def getIterBest(i):
    idx = np.argmax(accuracies[i])
    return [accuracies[i][idx],excludeds[i][idx]]

BEST = 0
THRESHOLD = 0.01
for a in accuracies:
    for b in a:
        if b > BEST:
            BEST = b

last_best = getIterBest(len(accuracies)-2)
max_iter = len(features_oc)
for j in range(len(accuracies)-1,max_iter):
    current_features = [f for f in features_oc if f not in last_best[1]]
    for i in range(len(accuracies[j]),len(current_features)):
        currently_excluded = current_features[i]
        props['features_vox'] = [f for f in current_features if f != currently_excluded]
        pickleSave(STATENAME,{'accuracies':accuracies,'excludeds':excludeds})
        ac = runModel(props)
        BEST = max([BEST,ac])
        accuracies[j].append(ac)
        excludeds[j].append(last_best[1]+[currently_excluded])
        logStatus(i,currently_excluded,ac)
    accuracies.append([])
    excludeds.append([])
    last_best = getIterBest(j)
    log('===================================')
    logStatus(0,'BEST',BEST)
    logStatus(j,last_best[1][-1],last_best[0])
    if BEST-THRESHOLD < last_best[0]:
        log('Stopping!')
        break
    log('===================================')