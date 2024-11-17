from DataHandler import DataHandler

handler = DataHandler(path='data/native', debug=True, out='console', cores=-1)
handler.preloadDataBase()
handler = DataHandler(path='data/normalized', debug=True, out='console', cores=-1)
handler.preloadDataBase()
