import sys
from DataHandler import DataHandler

kernelWidth=5
binWidth=25
absolute=True
inp='t1'
space='native'
if len(sys.argv) > 1:
    kernelWidth=int(sys.argv[1])
if len(sys.argv) > 2:
    binWidth=float(sys.argv[2])
    if binWidth % 1 == 0:
        binWidth = int(binWidth)
if len(sys.argv) > 3:
    tmp = sys.argv[3].lower()
    if tmp in ['false','true']:
        absolute = tmp == 'true'
    else:
        absolute='rel' not in tmp
if len(sys.argv) > 4:
    inp=sys.argv[4]
if len(sys.argv) > 5:
    space=sys.argv[5]
print('kernel_width={}, bin_width={}, absolute={}, inp={}, space={}'.format(kernelWidth,binWidth,absolute,inp,space))
binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
handler = DataHandler(
    path='data',
    space=space,
    clear_log=True,
    cores=7,
    out='logs/{}_{}_radiomics_k{}_b{}.log'.format(space,inp,kernelWidth,binstr),
)
handler.radiomicsVoxel(kernelWidth, binWidth, True, absolute, inp)
handler.deletePartialData(kernelWidth, binWidth, absolute, inp)
handler.scaleRadiomicsVoxel(kernelWidth, binWidth, absolute, inp)
handler.preloadRadiomicsVoxel(kernelWidth, binWidth, absolute, inp)