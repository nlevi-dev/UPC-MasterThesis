from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam

extras = {}
gen = DataGenerator(**props)
for names in gen.names:
    for n in names:
        extras[n] = []

for i in range(7):
    props['single'] = i
    gen = DataGenerator(**props)
    model = buildModel('FFN_chain_{}'.format(i))
    model.load_weights('data/models/FFN_chain_{}.weights.h5'.format(i))
    for names in gen.names:
        for n in names:
            extras[n].append(model.predict(gen.getReconstructor(n,x_only=True),0,verbose=False))