from DataHandler import DataHandler

# handler = DataHandler(path='data', space='native', out='tmp.log', clear_log=True, cores=-1)
# handler.preloadRadiomicsVoxel(5, 25)
# handler.preloadRadiomicsVoxel(7, 25)
# handler.preloadRadiomicsVoxel(9, 25)
# handler.preloadRadiomicsVoxel(11, 25)
# handler.preloadRadiomicsVoxel(13, 25)

handler = DataHandler(path='data', out='console', cores=1)
handler.register()
handler.normalize()