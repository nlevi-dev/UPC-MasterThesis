from DataHandler import DataHandler

handler = DataHandler(debug=True, out='l_radiomics_voxel.log', cores=6)
kernelWidth=5
binWidth=25
handler.radiomicsVoxel(kernelWidth, binWidth, recompute=True)
handler.deletePartialData(kernelWidth, binWidth)
handler.scaleRadiomics(kernelWidth, binWidth)