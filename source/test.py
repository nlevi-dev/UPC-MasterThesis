from DataHandler import DataHandler

handler = DataHandler(path='data',space='native',cores=8,out='console',visualize=True)
handler.scaleRadiomicsVoxel(5, 25, True, 't1')