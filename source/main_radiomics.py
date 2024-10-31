import sys
from DataHandler import DataHandler

binWidth=25
if len(sys.argv) > 1:
    binWidth=int(sys.argv[1])
print('bin_width={}'.format(binWidth))
handler = DataHandler(debug=True, out='main_radiomics.log', cores=-1)
handler.radiomics(binWidth, recompute=True)
handler.scaleRadiomics(binWidth)
handler.preloadData(binWidth)