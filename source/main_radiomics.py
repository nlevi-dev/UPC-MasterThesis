import sys
from DataHandler import DataHandler

binWidth=25
absolute=True
inp='t1'
space='native'
if len(sys.argv) > 1:
    binWidth=int(sys.argv[1])
if len(sys.argv) > 2:
    absolute='rel' not in sys.argv[2].lower()
if len(sys.argv) > 3:
    inp=sys.argv[3]
if len(sys.argv) > 4:
    space=sys.argv[4]
print('bin_width={}, absolute={}, inp={}, space={}'.format(binWidth,absolute,inp,space))
handler = DataHandler(path='data/'+space, out='logs/{}_{}_radiomics_b{}{}.log'.format(space,inp,binWidth,'' if absolute else 'r'), clear_log=True, cores=-1)
handler.radiomics(binWidth, absolute, inp)
handler.scaleRadiomics(binWidth, absolute, inp)
handler.preloadRadiomics(binWidth, absolute, inp)