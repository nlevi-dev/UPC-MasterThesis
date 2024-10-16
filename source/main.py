from DataHandler import DataHandler

handler = DataHandler(debug=True,out='log.txt',cores=lambda c:c-2)
# handler = DataHandler(debug=True,out='console',cores=1)
handler.preprocess()