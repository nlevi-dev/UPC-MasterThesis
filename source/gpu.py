import os
import subprocess

def getGpuMemoryUsage():
    ret = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell=True).decode('utf8')
    ret = ret.split('\n')
    if len(ret) <= 1:
        return []
    multiplier = 1.0
    if 'gib' in ret[0].lower():
        multiplier = 1024.0
    elif 'kib' in ret[0].lower():
        multiplier = 1.0/1024.0
    ret = ret[1:]
    ret = [r for r in ret if r.strip() != '']
    ret = [int(float(r[:r.find(' ')])*multiplier) for r in ret]
    return ret

mems = getGpuMemoryUsage()
for idx in range(len(mems)):
    if mems[idx] < 2048:
        os.environ['CUDA_VISIBLE_DEVICES']=str(idx)
        print('Using GPU {}!'.format(idx))
        break