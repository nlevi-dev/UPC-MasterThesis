import sys
from DataHandler import DataHandler

binWidth=25
absolute=True
inp='t1'
space='native'
if len(sys.argv) > 1:
    binWidth=float(sys.argv[1])
    if binWidth % 1 == 0:
        binWidth = int(binWidth)
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
binstr = str(binWidth).replace('.','')+('' if absolute else 'r')
handler = DataHandler(
    path='data',
    space=space,
    clear_log=True,
    cores=-1,
    out='logs/{}_{}_radiomics_b{}.log'.format(space,inp,binstr),
)
handler.radiomics(binWidth, absolute, inp)
handler.scaleRadiomics(binWidth, absolute, inp)
handler.preloadRadiomics(binWidth, absolute, inp)