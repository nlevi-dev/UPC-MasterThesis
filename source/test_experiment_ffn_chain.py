from model_ffn_chain import *
from DataGeneratorFFN import DataGenerator
from main_experiment_ffn_chain_body import props_override as props_override_chain_body
from main_experiment_ffn_chain_head import props_override as props_override_chain_head
from main_experiment_ffn_chain_regressor import props_override as props_override_chain_regressor

props = createProps(props_default, props_override_chain_body)

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
    name = 'FFN_chain_body_{}'.format(i)
    model = buildModel(x.shape[1],y.shape[1],name=name)
    model.load_weights('data/models/{}.weights.h5'.format(name))
    for n in extras.keys():
        extras[n].append(model.predict(gen.getReconstructor(n,xy_only=True)[0],0,verbose=False))
for n in extras.keys():
    extras[n] = np.concatenate(extras[n],-1)
#=============================================================================================================#

props = createProps(props_default, props_override_chain_head)

#======================================= collect predictions from head =======================================#
gen = DataGenerator(**props)
gen.extras = extras
train, val, test = gen.getData()

model = buildModel(train[0].shape[1],train[1].shape[1],name='FFN_chain_head')
model.load_weights('data/models/FFN_chain_head.weights.h5')
#=============================================================================================================#