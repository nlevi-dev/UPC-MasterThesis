from DataHandler import DataHandler

# handler = DataHandler(path='data', cores=16)
# handler.inverseWarp()
# handler = DataHandler(path='data', space='native', cores=16)
# handler.preprocess()
# handler.scaleTargets()
# handler.preloadTarget()
# handler = DataHandler(path='data', space='normalized', cores=16)
# handler.preprocess()
# handler.scaleTargets()
# handler.preloadTarget()
handler = DataHandler(path='data', cores=16)
handler.preloadCoords()