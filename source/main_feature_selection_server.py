import os
import time
import threading
from flask import Flask, Response, request
import numpy as np
from util import getHashId, pickleSave, pickleLoad

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

features_oc = np.load('data/preprocessed/features_vox.npy')
STATENAME = 'data/feature_selection_distributed_state.pkl'
LOGNAME = 'logs/feature_selection_distributed.log'

features_maxlen = max([len(f) for f in features_oc])
def log(msg):
    msg = str(msg)
    print(msg)
    with open(LOGNAME,'a') as log:
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

global tasks
global results
global cnt
tasks = []
results = []
cnt = 0
lock = threading.Lock()

def tasks_finish():
    global tasks
    global results
    global cnt
    while any([r is None for r in results]):
        time.sleep(60)
    res = results.copy()
    with lock:
        tasks = []
        results = []
        cnt = 0
    return res

def tasks_add(task):
    global tasks
    global results
    global cnt
    with lock:
        tasks.append(task)
        results.append(None)

def tasks_pop():
    global tasks
    global results
    global cnt
    if len(tasks) <= cnt:
        return None
    with lock:
        res = tasks[cnt]
        cnt += 1
        return res

def task_result(task,result):
    global tasks
    global results
    with lock:
        for i in range(len(tasks)):
            if tasks[i]['hashid'] == task['hashid']:
                results[i] = result
                logStatus(i,'BASELINE' if len(task['excluded'])==0 else task['excluded'][-1],result)
                break

def runModel(exc):
    if len(exc) == 0:
        props['features_vox'] = []
    else:
        props['features_vox'] = [f for f in features_oc if f not in exc]
    hashid = getHashId(architecture,props)[0]
    tasks_add({'excluded':exc,'hashid':hashid})

def producer():
    if os.path.exists(STATENAME):
        state = pickleLoad(STATENAME)
        accuracies = state['accuracies']
        excludeds = state['excludeds']
        BASELINE = accuracies[0][0]
    else:
        open(LOGNAME,'w').close()
        accuracies = []
        excludeds = []
        runModel([])
        results = tasks_finish()
        BASELINE = results[0]
        accuracies.append([BASELINE])
        accuracies.append([])
        excludeds.append([[]])
        excludeds.append([])
        pickleSave(STATENAME,{'accuracies':accuracies,'excludeds':excludeds})

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
        pickleSave(STATENAME,{'accuracies':accuracies,'excludeds':excludeds})
        for i in range(len(current_features)):
            exc = last_best[1]+[current_features[i]]
            runModel(exc)
            excludeds[j].append(exc)
        accuracies[j] = tasks_finish()
        BEST = max([BEST]+accuracies[j])
        accuracies.append([])
        excludeds.append([])
        last_best = getIterBest(j)
        log('==============================================')
        logStatus(0,'BEST',BEST)
        logStatus(j,last_best[1][-1],last_best[0])
        if BASELINE-THRESHOLD > last_best[0]:
            log('Stopping!')
            break
        log('==============================================')

app = Flask(__name__)

@app.route('/task_pop')
def consumer_handout():
    task = tasks_pop()
    if task is None:
        return Response('',status=503)
    return task

@app.route('/task_result', methods=['POST'])
def consumer_result():
    res = request.get_json()
    task_result(res['task'],res['result'])
    return Response('',status=200)

def consumer():
    app.run(
        host='0.0.0.0',
        port=15000,
        debug=False,
        use_reloader=False
    )

if __name__ == "__main__":
    prod = threading.Thread(target=producer)
    cons = threading.Thread(target=consumer)
    prod.start()
    cons.start()
    prod.join()