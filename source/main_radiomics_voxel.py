from DataHandler import DataHandler

handler = DataHandler(debug=True, out='l_radiomics_voxel.log', cores=6)
handler.radiomicsVoxel(kernelWidth=5, binWidth=25, recompute=True)
