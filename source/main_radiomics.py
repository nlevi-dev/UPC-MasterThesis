from DataHandler import DataHandler

handler = DataHandler(debug=True, out='l_radiomics.log', cores=-1)
handler.radiomics(binWidth=25)
