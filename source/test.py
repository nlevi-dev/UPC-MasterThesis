from DataHandler import DataHandler

handler = DataHandler(space='native')
handler.scaleTargets()
handler.preloadTarget()
handler = DataHandler(space='normalized')
handler.scaleTargets()
handler.preloadTarget()