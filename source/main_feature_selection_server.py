import os
import time
import threading
import numpy as np
from util import getHashId, pickleSave, pickleLoad
if int(os.environ.get('MINIMAL','0'))<2:
    from flask import Flask, Response, request, send_from_directory

props={
    'path'          : 'data',
    'seed'          : 42,
    'split'         : 0.8,
    'test_split'    : 0.5,
    'control'       : True,
    'huntington'    : False,
    'left'          : True,
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
STATENAME = 'data/feature_selection.pkl'
LOGNAME = 'logs/feature_selection.log'
TIMEOUT = 150

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

global accuracies
global excludeds

global tasks
global results
global popped
tasks = []
results = []
popped = []
lock = threading.Lock()

def save_state():
    pickleSave(STATENAME,{
        'accuracies':accuracies,
        'excludeds':excludeds,
        'tasks':tasks,
        'results':results,
        'popped':popped,
    })

def tasks_finish():
    global tasks
    global results
    global popped
    while any([r is None for r in results]):
        time.sleep(60)
    res = results.copy()
    with lock:
        tasks = []
        results = []
        popped = []
    return res

def tasks_set(tasks_):
    global tasks
    global results
    global popped
    with lock:
        tasks = tasks_
        results = [None for _ in range(len(tasks_))]
        popped = [None for _ in range(len(tasks_))]
        save_state()

def tasks_pop(force=False):
    global tasks
    global results
    global popped
    idx = -1
    if force:
        for i in range(len(results)):
            if results[i] is None:
                idx = i
                break
        if idx == -1:
            return None
        with lock:
            for i in range(len(results)):
                if results[i] is None:
                    #sanity check
                    if idx != i:
                        return None
                    idx = i
                    break
            res = tasks[idx]
            popped[idx] = time.time()
            save_state()
            return res
    else:
        for i in range(len(popped)):
            if popped[i] is None:
                idx = i
                break
        if idx == -1:
            return None
        with lock:
            for i in range(len(popped)):
                if popped[i] is None:
                    #sanity check
                    if idx != i:
                        return None
                    idx = i
                    break
            res = tasks[idx]
            popped[idx] = time.time()
            save_state()
            return res

def tasks_keepalive(task):
    global tasks
    global popped
    with lock:
        for i in range(len(tasks)):
            if tasks[i]['hashid'] == task['hashid']:
                popped[i] = time.time()
                break

def tasks_timeout():
    global tasks
    global results
    global popped
    with lock:
        t = time.time()
        for i in range(len(popped)):
            if results[i] is None and popped[i] is not None and t-popped[i] > TIMEOUT:
                print('Timed out {}!'.format(tasks[i]))
                popped[i] = None
        save_state()

def tasks_result(task,result):
    global tasks
    global results
    with lock:
        for i in range(len(tasks)):
            if tasks[i]['hashid'] == task['hashid']:
                results[i] = result
                logStatus(i,'BASELINE' if len(task['excluded'])==0 else task['excluded'][-1],result)
                break

def producer():
    global accuracies
    global excludeds
    global tasks
    global results
    global popped
    if os.path.exists(STATENAME):
        state = pickleLoad(STATENAME)
        accuracies = state['accuracies']
        excludeds = state['excludeds']
        tasks = state['tasks']
        results = state['results']
        popped = state['popped']
        BASELINE = accuracies[0][0]
        for idx in [59,43,65]:
            results[idx] = None
            popped[idx] = None
        save_state()
    else:
        open(LOGNAME,'w').close()
        accuracies = []
        excludeds = []
        tasks_set([{'excluded':[],'hashid':getHashId(architecture,props)[0]}])
        results = tasks_finish()
        BASELINE = results[0]
        accuracies.append([BASELINE])
        accuracies.append([])
        excludeds.append([[]])
        excludeds.append([])
        save_state()

    def getIterBest(i):
        idx = np.argmax(accuracies[i])
        return [accuracies[i][idx],excludeds[i][idx]]

    BEST = 0
    THRESHOLD = 0.02
    for a in accuracies:
        for b in a:
            if b > BEST:
                BEST = b

    last_best = getIterBest(len(accuracies)-2)
    max_iter = len(features_oc)
    for j in range(len(accuracies)-1,max_iter):
        save_state()
        current_features = [f for f in features_oc if f not in last_best[1]]
        if len(tasks) == 0:
            ts = []
            for i in range(len(current_features)):
                exc = last_best[1]+[current_features[i]]
                excludeds[j].append(exc)
                props['features_vox'] = [f for f in features_oc if f not in exc]
                hashid = getHashId(architecture,props)[0]
                ts.append({'excluded':exc,'hashid':hashid})
            tasks_set(ts)
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

def timeout():
    while True:
        time.sleep(TIMEOUT)
        tasks_timeout()

if __name__ == "__main__":
    app = Flask(__name__)

    @app.route('/task_pop/<instance>', methods=['GET'])
    def consumer_pop(instance):
        task = tasks_pop()
        if task is None and 'H100' in instance:
            task = tasks_pop(force=True)
        if task is None:
            return Response('',status=503)
        return task

    @app.route('/task_result/<instance>', methods=['POST'])
    def consumer_result(instance):
        res = request.get_json()
        tasks_result(res['task'],res['result'])
        return Response('',status=200)

    @app.route('/task_keepalive/<instance>', methods=['POST'])
    def consumer_keepalive(instance):
        res = request.get_json()
        tasks_keepalive(res['task'])
        return Response('',status=200)

    @app.route('/download/data.zip', methods=['GET'])
    def download_data():
        return send_from_directory('','data.zip')

    @app.route('/download/source.zip', methods=['GET'])
    def download_source():
        return send_from_directory('','source.zip')

    @app.route('/upload/<name>', methods=['POST'])
    def upload(name):
        f = request.files['file']
        if '/' in name or '\\' in name or '..' in name:
            raise Exception('Invalid name!')
        if os.path.exists('data/models/'+name):
            os.remove('data/models/'+name)
        f.save('data/models/'+name)
        return Response('',status=200)

    def consumer():
        app.run(
            host='0.0.0.0',
            port=15000,
            debug=False,
            use_reloader=False
        )

    prod = threading.Thread(target=producer)
    tout = threading.Thread(target=timeout)
    cons = threading.Thread(target=consumer)
    prod.start()
    tout.start()
    cons.start()
    prod.join()