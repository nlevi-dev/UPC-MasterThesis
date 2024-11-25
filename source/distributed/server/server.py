import os
import json
import math
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from flask import send_from_directory, Flask, request, make_response
import threading
import time

meta = 'backend_version: alpha_v1.1\nmodel_type: YOLOv8\nmodel_version: v1.0\nmodel_epochs: 400\nmodel_training_size: 132\nmodel_labels_distribution: 65 263 0 0\nmodel_box_loss: 0.45\nmodel_cls_loss: 0.28'

IS_PRODUCTION = False
try:
    IS_PRODUCTION = os.environ['IS_PRODUCTION'] == 'true'
except:
    pass

model = YOLO('weights.pt')
print(model.info())

path = '/app/files' if IS_PRODUCTION else '/home/levente/nfs/public/materials/models/cricks/raw'

paths = []
latest_detected = ''
for root, _, files in os.walk(path):
    if root != path: continue
    for file in sorted(files):
        if file[0:9] == 'detected_': latest_detected = file
        if file[0] != '2': continue
        paths.append(file)
latest = paths[-1]

app = Flask(__name__)

@app.route('/latest', methods=['GET'])
def getLatest():
    response = make_response(send_from_directory(path,latest_detected))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
    return response

@app.route('/stats', methods=['GET'])
def getStats():
    response = make_response(send_from_directory(path,'detected.txt'))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/meta', methods=['GET'])
def getMeta():
    response = make_response(meta,200)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/upload', methods=['POST'])
def postUpload():
    f = request.files['image']
    f.save(path+'/'+datetime.now().strftime("%y-%m-%d_%H:%M:%S")+'.png')
    return make_response("OK", 200)

def flaskThread():
    app.run(port=3001,debug=False,host='0.0.0.0' if IS_PRODUCTION else '127.0.0.1')

threading.Thread(target=flaskThread, daemon=True).start()

def checkNew():
    global latest_detected
    paths = []
    for root, _, files in os.walk(path):
        if root != path: continue
        for file in sorted(files):
            if file[0:9] == 'detected_': latest_detected = file
            if file[0] != '2': continue
            paths.append(file)
    latest = paths[-1]

    if latest_detected[9:] != latest and IS_PRODUCTION:
        p = model.predict(path+'/'+latest, imgsz=640, conf=0.5)[0]
        line = str(round(datetime.timestamp(datetime.strptime(latest[0:17], '%y-%m-%d_%H:%M:%S'))))+' '
        cnt = np.unique(p.boxes.cls.cpu(), return_counts=True)
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        for j in range(len(cnt[0])):
            if cnt[0][j] == 0:
                c1 = cnt[1][j]
            elif cnt[0][j] == 1:
                c2 = cnt[1][j]
            elif cnt[0][j] == 2:
                c3 = cnt[1][j]
            elif cnt[0][j] == 3:
                c4 = cnt[1][j]
        line += str(round(c1))+' '
        line += str(round(c2))+' '
        line += str(round(c3))+' '
        line += str(round(c4))+'\n'
        os.remove(path+'/'+latest_detected)
        latest_detected = 'detected_'+latest
        cv2.imwrite(path+'/'+latest_detected,p.plot(line_width=1))
        del p
        file = open(path+'/detected.txt','a')
        file.write(line)
        file.close()

while True:
    checkNew()
    time.sleep(60 if IS_PRODUCTION else 3)
