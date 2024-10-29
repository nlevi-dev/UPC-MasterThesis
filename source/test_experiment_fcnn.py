from model_fcnn import *

model = buildModel()
model.load_weights('data/models/FCNN.weights.h5')

# plotModel(model)

showResults(model,threshold=0,background=False)