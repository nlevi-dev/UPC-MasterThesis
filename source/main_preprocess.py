from DataHandler import DataHandler

handler = DataHandler(path='data', out='logs/preprocess.log', clear_log=True, cores=-1)
handler.register()
handler.normalize()
handler = DataHandler(path='data', space='native', out='logs/preprocess.log', clear_log=False, cores=-1)
handler.preprocess()
handler.preloadTarget()
handler = DataHandler(path='data', space='normalized', out='logs/preprocess.log', clear_log=False, cores=-1)
handler.preprocess()
handler.preloadTarget()