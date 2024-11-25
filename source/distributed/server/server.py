import os
from datetime import datetime
from flask import Flask, request, make_response

app = Flask(__name__)

os.makedirs('mount/logs',exist_ok=True)

@app.route('/upload/<name>', methods=['POST'])
def postUpload(name):
    f = request.files['result']
    name = name.replace('/','').replace('\\','')
    f.save('mount/'+name+'_'+datetime.now().strftime("%y-%m-%d_%H-%M-%S")+'.zip')
    return make_response("OK", 200)

@app.route('/log/<name>', methods=['POST'])
def postLog(name):
    f = request.files['log']
    name = name.replace('/','').replace('\\','')
    if os.path.exists('mount/logs/'+name):
        os.remove('mount/logs/'+name)
    f.save('mount/logs/'+name)
    return make_response("OK", 200)

app.run(port=3001,debug=False,host='0.0.0.0')