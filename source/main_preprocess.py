from DataHandler import DataHandler

handler = DataHandler(path='data', out='logs/preprocess.log', clear_log=True, cores=-1)
handler.register()
handler.normalize()
handler = DataHandler(path='data/native', out='logs/preprocess.log', clear_log=False, cores=-1)
handler.preprocess(crop_to_bounds=True)
handler = DataHandler(path='data/normalized', out='logs/preprocess.log', clear_log=False, cores=-1)
handler.preprocess(crop_to_bounds=False)