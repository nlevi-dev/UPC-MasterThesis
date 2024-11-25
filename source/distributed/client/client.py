import os
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
DEBUG = os.environ.get('DEBUG','false').lower() == 'true'
NAME = os.environ.get('NAME','default')

#computation
print('kernel_width={}, bin_width={}, absolute={}, inp={}, space={}'.format(kernelWidth,binWidth,absolute,inp,space))
handler = DataHandler(
    path='mount/data',
    space=space,
    clear_log=False,
    cores=cores,
    out='mount/{}_{}_{}_radiomics_k{}_b{}{}.log'.format(NAME,space,inp,kernelWidth,binWidth,'' if absolute else 'r'),
    partial=range(0,6) if DEBUG else None,
)
handler.radiomicsVoxel(kernelWidth, binWidth, False, absolute, inp, fastOnly=DEBUG)
handler.deletePartialData(kernelWidth, binWidth, absolute, inp)
