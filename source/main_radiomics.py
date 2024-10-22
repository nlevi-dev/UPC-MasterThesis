from DataHandler import DataHandler

kernelWidth=5
binWidth=25
handler = DataHandler(debug=True, out='l_radiomics_voxel.log', cores=-1)
handler.radiomics(binWidth)
handler.cores = 6
handler.radiomicsVoxel(kernelWidth, binWidth, recompute=True)
handler.deletePartialData(kernelWidth, binWidth)
handler.scaleRadiomics(kernelWidth, binWidth)