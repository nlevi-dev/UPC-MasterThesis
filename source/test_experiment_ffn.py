from model_ffn import *
from DataGeneratorFFN import DataGenerator

props['debug'] = True
gen = DataGenerator(**props)
train, val, test = gen.getData()

model = buildModel(train[0].shape[-1])
model.load_weights('data/models/FFN.weights.h5')

showResults(model,gen,background=True)
