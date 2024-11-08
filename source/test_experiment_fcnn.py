from model_fcnn import *
from DataGeneratorFCNN import DataGenerator

props['debug'] = True
gen = DataGenerator(**props)
train, val, test = gen.getData()

model = buildModel(train[0].shape[1:])
model.load_weights('data/models/FCNN.weights.h5')

showResults(model,gen,background=True)
