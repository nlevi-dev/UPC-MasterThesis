import numpy as np
from util import pickleLoad

feature_selection = pickleLoad('data/feature_selection.pkl')
features_oc = np.load('data/preprocessed/features_vox.npy')
exc = feature_selection['excludeds']
acc = feature_selection['accuracies']
tmp = []
for i in range(1,len(exc)-1):
    idx = np.argmax(acc[i])
    for f in exc[i][idx]:
        if f not in tmp:
            print('{}. & {} & {} \\\\ \\hline'.format(i,f.replace('_','\\_'),round(acc[i][idx]*100,1)))
            break
    tmp = exc[i][idx]