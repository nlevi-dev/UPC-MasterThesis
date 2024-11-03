from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam
from DataGeneratorFFN import DataGenerator
from main_experiment_ffn_chain_body import props_override as props_override_chain_body

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
    name = 'FFN_chain_{}'.format(i)
    model = buildModel(x.shape[1],y.shape[1],name=name)
    model.load_weights('data/models/{}.weights.h5'.format(name))
    for n in extras.keys():
        extras[n].append(model.predict(gen.getReconstructor(n,xy_only=True)[0],0,verbose=False))
for n in extras.keys():
    extras[n] = np.concatenate(extras[n],-1)
#=============================================================================================================#

props_override = {
    'threshold'     : 0,          #one hot encode
    'binarize'      : True,       #one hot encode
    'single'        : None,       #all layers (classification model)
    'radiomics'     : ['b25'],    #simple model
    'radiomics_vox' : ['k5_b25'], #simple model
    'balance_data'  : False,      #no balanced data
}
props = createProps(props_default, props_override)   

batch_size = 100000

gen = DataGenerator(**props)
gen.extras = extras
train, val, test = gen.getData()

name = 'FFN_chain_head'
print(name)

optimizer = Adam(learning_rate=0.001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/{}.weights.h5'.format(name),monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel(train[0].shape[1],train[1].shape[1],name=name)

model.compile(loss=CCE, optimizer='adam', jit_compile=True, metrics=[MMAE])

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=10000,
    verbose=1,
    callbacks = [save,stop],
)

showResults(model, gen, threshold=0)