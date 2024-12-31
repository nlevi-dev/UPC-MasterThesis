import os
import time
from datetime import datetime
import threading
import subprocess
from RecordHandler import RecordHandler

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

#name of log file
outname = '{}_{}_{}_radiomics_k{}_b{}{}.log'.format(NAME,space,inp,kernelWidth,binWidth,'' if absolute else 'r')

#log uploader
def uploadLog():
    subprocess.call('curl -X POST -F log=@mount/{} "$ADDRESS/log/{}"'.format(outname,outname), shell=True)
global RUN
RUN = True
sleepsegment=60
def uploadLogWrapper():
    time.sleep(10)
    uploadLog()
    tmp0 = datetime.now()
    tmp1 = tmp0.replace(hour=tmp0.hour+1,minute=0,second=0,microsecond=0)
    tmp0 = tmp0.timestamp()
    tmp1 = tmp1.timestamp()
    global RUN
    sleepfor=int(tmp1-tmp0)
    time.sleep(sleepfor%sleepsegment)
    for _ in range(sleepfor//sleepsegment):
        if not RUN:
            break
        time.sleep(sleepsegment)
    sleepfor=3600
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
handler = RecordHandler(
    path='mount/data',
    space=space,
    clear_log=False,
    cores=cores,
    out='mount/'+outname,
    partial=range(0,6) if DEBUG else None,
)
handler.radiomicsVoxel(kernelWidth, binWidth, False, absolute, inp, fastOnly=DEBUG)
handler.deletePartialData(kernelWidth, binWidth, absolute, inp)

#final log upload
RUN = False
uploadLog()