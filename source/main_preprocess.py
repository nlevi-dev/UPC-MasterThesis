from DataHandler import DataHandler

handler = DataHandler(debug=True, out='main_preprocess.log', cores=-1)
handler.preprocess()
handler.preloadDataConnection()