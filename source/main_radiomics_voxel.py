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
    binWidth=int(sys.argv[2])
if len(sys.argv) > 3:
    absolute='rel' not in sys.argv[3].lower()
if len(sys.argv) > 4:
    inp=sys.argv[4]
if len(sys.argv) > 5:
    space=sys.argv[5])
handler = DataHandler(path='data/'+space, names='names2' if inp == 't1t2' else 'names1', clear_log=True, cores=-1, out='logs/{}_{}_radiomics_k{}_b{}{}.log'.format(space,inp,kernelWidth,binWidth,'' if absolute else 'r'))
handler.radiomicsVoxel(kernelWidth, binWidth, True, absolute, inp)
handler.deletePartialData(kernelWidth, binWidth, absolute, inp)
handler.scaleRadiomicsVoxel(kernelWidth, binWidth, absolute, inp)
handler.preloadRadiomicsVoxel(kernelWidth, binWidth, absolute, inp)