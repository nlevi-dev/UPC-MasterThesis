from DataHandler import DataHandler

import multiprocessing
bfg = multiprocessing.cpu_count() > 8
p = (0.35,1) if bfg else (0,0.35)
c = 6 if bfg else 4

handler = DataHandler(debug=True, out='l_radiomics_voxel.log', cores=c, partial=p)
handler.radiomicsVoxel(kernelWidth=5, binWidth=25, recompute=True)
