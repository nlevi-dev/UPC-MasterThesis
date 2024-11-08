from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam
from DataGeneratorFFN import DataGenerator
from main_experiment_ffn_chain_body import props_override as props_override_chain_body

props_override = {
    'threshold'     : None,  #no threshold
    'binarize'      : False, #no binarize
    'single'        : None,  #no single
    'radiomics'     : props_override_chain_body['radiomics'],    #same complex model
    'radiomics_vox' : props_override_chain_body['radiomics_vox'],#same complex model
    'balance_data'  : False, #no balance
    'collapse_max'  : True,  #collapse max
}
props = createProps(props_default, props_override)

batch_size = 10000

gen = DataGenerator(**props)
train, val, test = gen.getData()

name = 'FFN_chain_regressor'
print(name)

optimizer = Adam(learning_rate=0.001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/{}.weights.h5'.format(name),monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel(train[0].shape[1],1,name=name)

model.compile(loss=MSE, optimizer='adam', jit_compile=True, metrics=[MAE,STD])

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=10000,
    verbose=1,
    callbacks = [save,stop],
)