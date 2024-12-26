import os
import sys
import json
import numpy as np
import pandas as pd

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

pd.set_option('display.width',os.get_terminal_size().columns)

def getNotebooks(path='.'):
    raws = os.listdir(path)
    raws = sorted(raws)
    raws = [path+'/'+r for r in raws]
    dirs = [r for r in raws if not os.path.isfile(r)]
    fils = [r for r in raws if os.path.isfile(r)]
    for d in dirs:
        try:
            fils += getNotebooks(d)
        except:
            pass
    fils = [f for f in fils if f[-6:] == '.ipynb']
    return fils

notebooks = getNotebooks()

results = []

for n in notebooks:
    try:
        res = [n[2:-6],'','','','','','','','','','']
        with open(n,'r') as f:
            data = json.load(f)
        for c in data['cells']:
            if c['cell_type'] != 'code': continue
            result = []
            for o in c['outputs']:
                if 'name' in o.keys() and o['name'] == 'stdout':
                    result += o['text']
            if 'from DataGenerator import DataGenerator' in ''.join(c['source']):
                res[10] = result[1][result[1].index(', ')+2:result[1].index(')')]
            if 'getAccuracy' not in ''.join(c['source']) and 'getPearson' not in ''.join(c['source']): continue
            if 'train' in result[0]:
                res[1] = round(float(result[1].strip())*100,1)
                res[2] = round(float(result[2].strip())*100,1)
                res[3] = round(float(result[3].strip())*100,1)
            elif 'native' in result[0]:
                res[4] = round(float(result[1].strip())*100,1)
                res[5] = round(float(result[2].strip())*100,1)
                res[6] = round(float(result[3].strip())*100,1)
            elif 'normalized' in result[0]:
                res[7] = round(float(result[1].strip())*100,1)
                res[8] = round(float(result[2].strip())*100,1)
                res[9] = round(float(result[3].strip())*100,1)
        results.append(res)
    except:
        pass

columns = ['experiment','tra_train','tra_val','tra_test','nat_train','nat_val','nat_test','nor_train','nor_val','nor_test']
w = max([len(c) for c in columns[1:]])
for i in range(1,len(columns)):
    while len(columns[i]) < w:
        columns[i] += ' '

results = np.array(results)

if len(sys.argv) > 2 and sys.argv[2] == 'latex':
    s = ''
    for r in results:
        n = r[0][r[0].index('/')+1:]
        n1 = int(n[:n.index('-')])
        n2 = n[n.index('-')+1:]
        s += '{}. & \\textbf{} & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ \\\\ \\hline\n'.format(n1,'{'+n2.replace('_','\\_')+'}',r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10])
    print(s)
else:
    results = results[:,:-1]
    df = pd.DataFrame(results, columns=columns)
    def ljust(s):
        s = s.astype(str).str.strip()
        return s.str.ljust(s.str.len().max())
    print(df.apply(ljust).to_string(index=False,justify='left'))