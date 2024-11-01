from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam

#======================================= collect predictions from body =======================================#
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
    for n in extras.keys():
        extras[n].append(model.predict(gen.getReconstructor(n,x_only=True),0,verbose=False))
for n in extras.keys():
    extras[n] = np.concatenate(extras[n],-1)
#=============================================================================================================#

batch_size = 100000
props['radiomics'] = ['b25']
props['radiomics_vox'] = ['k5_b25','k11_b25']
props['targets_all'] = True

for i in range(7):
    props['single'] = i
    gen = DataGenerator(**props)
    gen.extras = extras
    train, val, test = gen.getData()

    name = 'FFN_chain_{}_head'.format(i)
    print(name)

    optimizer = Adam(learning_rate=0.001)
    stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
    save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/{}.weights.h5'.format(name),monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

    model = buildModel(train[0].shape[1],1,name=name)

    model.compile(loss=BCE, optimizer='adam', jit_compile=True, metrics=[MAE])

    history = model.fit(DataWrapper(train,batch_size),
        validation_data=DataWrapper(val,batch_size,False),
        epochs=10000,
        verbose=1,
        callbacks = [save,stop],
    )