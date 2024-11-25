import os
import time
import threading
import subprocess
from DataHandler import DataHandler

#load env variables
kernelWidth = int(os.environ.get('kernelWidth', '5'))
binWidth = int(os.environ.get('binWidth', '25'))
absolute = os.environ.get('absolute', 'abs')
tmp = absolute.lower()
if tmp in ['false','true']:
    absolute = tmp == 'true'
else:
    absolute='rel' not in tmp
inp = os.environ.get('inp', 't1')
space = os.environ.get('space', 'native')
cores = int(os.environ.get('cores', '6'))
DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
NAME = os.environ.get('NAME', 'UNDEFINED')

outname = '{}_{}_{}_radiomics_k{}_b{}{}.log'.format(NAME,space,inp,kernelWidth,binWidth,'' if absolute else 'r')

def uploadLog():
    subprocess.call('curl -X POST -F log=@mount/{} "$ADDRESS/log/{}"'.format(outname,outname), shell=True)
global RUN
RUN = True
sleepfor=3600
sleepsegment=60
def uploadLogWrapper():
    global RUN
    while RUN:
        uploadLog()
        for _ in range(sleepfor//sleepsegment):
            if not RUN:
                break
            time.sleep(sleepsegment)
uploader = threading.Thread(target=uploadLogWrapper)
uploader.start()

#computation
print('kernel_width={}, bin_width={}, absolute={}, inp={}, space={}'.format(kernelWidth,binWidth,absolute,inp,space))
handler = DataHandler(
    path='mount/data',
    space=space,
    clear_log=False,
    cores=cores,
    out='mount/'+outname,
    partial=range(0,6) if DEBUG else None,
)
ran = handler.radiomicsVoxel(kernelWidth, binWidth, False, absolute, inp, fastOnly=DEBUG)
if ran:
    handler.deletePartialData(kernelWidth, binWidth, absolute, inp)

RUN = False
uploadLog()