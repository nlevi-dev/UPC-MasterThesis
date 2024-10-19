from DataHandler import DataHandler

import multiprocessing
bfg = multiprocessing.cpu_count() > 8
p = (0.35,1) if bfg else (0,0.35)
c = 6 if bfg else 4
o = 'l_radiomics_voxel_bfg.log' if bfg else 'l_radiomics_voxel_lap.log'

handler = DataHandler(debug=True, out=o, cores=c, partial=p)
handler.radiomicsVoxel(kernelWidth=5, binWidth=25, recompute=False)
