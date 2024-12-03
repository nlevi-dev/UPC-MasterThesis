import os
import json
import numpy as np
import pandas as pd

pd.set_option('display.width',os.get_terminal_size().columns)

def getNotebooks(path='.'):
    raws = os.listdir(path)
    raws = sorted(raws)
    raws = [path+'/'+r for r in raws]
    dirs = [r for r in raws if not os.path.isfile(r)]
    fils = [r for r in raws if os.path.isfile(r)]
    for d in dirs:
        fils += getNotebooks(d)
    fils = [f for f in fils if f[-6:] == '.ipynb']
    return fils

notebooks = getNotebooks()

results = []

for n in notebooks:
    res = [n[2:-6],'','','','','','']
    with open(n,'r') as f:
        data = json.load(f)
    for c in data['cells']:
        if c['cell_type'] != 'code': continue
        if 'getAccuarcy' not in ''.join(c['source']): continue
        result = []
        for o in c['outputs']:
            if o['name'] == 'stdout':
                result += o['text']
        if 'balanced' in result[0]:
            res[4] = round(float(result[1].strip())*100,1)
            res[5] = round(float(result[2].strip())*100,1)
            res[6] = round(float(result[3].strip())*100,1)
        else:
            idx = len(result)-3
            res[1] = round(float(result[idx+0].strip())*100,1)
            res[2] = round(float(result[idx+1].strip())*100,1)
            res[3] = round(float(result[idx+2].strip())*100,1)
    results.append(res)

results = np.array(results)
df = pd.DataFrame(results, columns=['experiment','train acc','val acc','test acc','bal train acc','bal val acc','bal test acc'])
def ljust(s):
    s = s.astype(str).str.strip()
    return s.str.ljust(s.str.len().max())
print(df.apply(ljust).to_string(index=False,justify='left'))