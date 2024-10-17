from DataHandler import DataHandler

handler = DataHandler(debug=True, out='l2_radiomics_voxel.log', cores=0.5, partial=(0.5,1))
handler.radiomicsVoxel(kernelWidth=5, binWidth=25, forceReCompute=True, excludeSlow=False)
