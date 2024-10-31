import sys
from DataHandler import DataHandler

kernelWidth=5
binWidth=25
if len(sys.argv) > 1:
    kernelWidth=int(sys.argv[1])
    binWidth=int(sys.argv[2])
print('kernel_width={},bin_width={}'.format(kernelWidth,binWidth))
handler = DataHandler(debug=True, out='main_radiomics_voxel.log', cores=6)
handler.radiomicsVoxel(kernelWidth, binWidth, recompute=True)
handler.deletePartialData(kernelWidth, binWidth)
handler.scaleRadiomicsVoxel(kernelWidth, binWidth)
handler.preloadDataVoxel(kernelWidth, binWidth)