from DataHandler import DataHandler

handler = DataHandler(path='data',space='native',cores=8,out='console',visualize=True)
handler.preloadRadiomicsVoxel(5, 25, True, 't1')