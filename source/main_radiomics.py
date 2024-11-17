import sys
from DataHandler import DataHandler

binWidth=25
absolute=True
inp='t1'
space='native'
if len(sys.argv) > 1:
    binWidth=int(sys.argv[1])
if len(sys.argv) > 2:
    tmp = sys.argv[2].lower()
    if tmp in ['false','true']:
        absolute = tmp == 'true'
    else:
        absolute='rel' not in tmp
if len(sys.argv) > 3:
    inp=sys.argv[3]
if len(sys.argv) > 4:
    space=sys.argv[4]
print('bin_width={}, absolute={}, inp={}, space={}'.format(binWidth,absolute,inp,space))
handler = DataHandler(path='data/'+space, names='names2' if inp == 't1t2' else 'names1', clear_log=True, cores=-1, out='logs/{}_{}_radiomics_b{}{}.log'.format(space,inp,binWidth,'' if absolute else 'r'))
handler.radiomics(binWidth, absolute, inp)
handler.scaleRadiomics(binWidth, absolute, inp)
handler.preloadRadiomics(binWidth, absolute, inp)