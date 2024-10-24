import sys
from DataHandler import DataHandler

kernelWidth=5
binWidth=25
if len(sys.argv) > 1:
    kernelWidth=int(sys.argv[1])
    binWidth=int(sys.argv[2])
print('kernel_width={},bin_width={}'.format(kernelWidth,binWidth))
handler = DataHandler(debug=True, out='main_radiomics.log', cores=-1)
handler.radiomics(binWidth)
handler.cores = 6
handler.radiomicsVoxel(kernelWidth, binWidth, recompute=True)
handler.deletePartialData(kernelWidth, binWidth)
handler.scaleRadiomics(kernelWidth, binWidth)
handler.preloadData(kernelWidth, binWidth)