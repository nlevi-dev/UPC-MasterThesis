from model_ffn_chain import *
from DataGeneratorFFN import DataGenerator

#======================================= collect predictions from body =======================================#
extras = {}
gen = DataGenerator(**props)
for names in gen.names:
    for n in names:
        extras[n] = []
for i in range(7):
    props['single'] = i
    gen = DataGenerator(**props)
    x, y = gen.getReconstructor(gen.names[0][0],xy_only=True)
    name = 'FFN_chain_{}'.format(i)
    model = buildModel(x.shape[1],y.shape[1],name=name)
    model.load_weights('data/models/{}.weights.h5'.format(name))
    for n in extras.keys():
        extras[n].append(model.predict(gen.getReconstructor(n,xy_only=True)[0],0,verbose=False))
for n in extras.keys():
    extras[n] = np.concatenate(extras[n],-1)
#=============================================================================================================#

props['single'] = None
props['not_connected'] = True
props['radiomics'] = ['b25']
props['radiomics_vox'] = ['k5_b25','k11_b25']

gen = DataGenerator(**props)
gen.extras = extras
train, val, test = gen.getData()

model = buildModel(train[0].shape[1],train[1].shape[1],name='FFN_chain_head')
model.load_weights('data/models/FFN_chain_head.weights.h5')

showResults(model, gen, threshold=0)