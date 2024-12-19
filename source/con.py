import numpy as np

con = np.load('data/native/preloaded/C01_1/connectivity_left.npy')
ste = np.load('data/native/preloaded/C01_1/streamline_left.npy')

print(np.max(con))
print(np.max(ste))

def ste2con(ste, threshold=0):
    ste = np.where(ste < threshold, 0, ste)
    ste2 = np.repeat(np.expand_dims(np.sum(ste,-1),-1),14,-1)
    ste2 = np.where(ste2==0,1,ste2)
    ste2 = ste/ste2
    return ste2

for t in range(10):
    thr = t*100
    print(thr)
    thr = thr/5000.0
    ste2 = ste2con(ste, thr)
    dif = con-ste2
    print(np.min(dif))
    # print(np.max(dif))
    # print(np.mean(dif))