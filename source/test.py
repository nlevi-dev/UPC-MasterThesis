from DataHandler import DataHandler

# handler = DataHandler(path='data', space='native', cores=12)
# handler.scaleTargets()
# handler.preloadTarget()
# handler = DataHandler(path='data', space='normalized', cores=12)
# handler.scaleTargets()
# handler.preloadTarget()
handler = DataHandler(path='data', cores=12)
handler.processClinical()