from DataHandler import DataHandler

handler = DataHandler(path='data/native', debug=True, out='console', cores=-1)
handler.preloadRadiomicsVoxel(5, 25)
handler.preloadRadiomicsVoxel(7, 25)
handler.preloadRadiomicsVoxel(9, 25)
handler.preloadRadiomicsVoxel(11, 25)
handler.preloadRadiomicsVoxel(13, 25)
handler.preloadRadiomics(10)
handler.preloadRadiomics(25)
handler.preloadRadiomics(50)
handler.preloadRadiomics(75)