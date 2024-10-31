from model_ffn import *

model = buildModel()
model.load_weights('data/models/FFN.weights.h5')

model.summary()

plotModel(model)

showResults(model,threshold=0,background=True)
