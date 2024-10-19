from DataHandler import DataHandler

import multiprocessing; p = (0,0.5) if multiprocessing.cpu_count() > 8 else (0.5,1)

handler = DataHandler(debug=True, out='l_radiomics_voxel.log', cores=6, partial=p)
handler.radiomicsVoxel(kernelWidth=5, binWidth=25, recompute=False)
