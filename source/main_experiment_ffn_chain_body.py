from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam
from DataGeneratorFFN import DataGenerator

props_override = {
    'threshold'     : 0,    #one hot encode
    'binarize'      : True, #one hot encode
    'single'        : 0,    #one model per layer
    'radiomics'     : ['b10','b25','b50','b75'],              #complex model
    'radiomics_vox' : ['k5_b25','k7_b25','k9_b25','k11_b25'], #complex model
    'balance_data'  : True, #balanced data
}

batch_size = 100000

for i in range(7):
    props_override['single'] = i
    props = createProps(props_default, props_override)
    gen = DataGenerator(**props)
    train, val, test = gen.getData()

    name = 'FFN_chain_body_{}'.format(i)
    print(name)

    optimizer = Adam(learning_rate=0.001)
    stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
    save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/{}.weights.h5'.format(name),monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

    model = buildModel(train[0].shape[1],1,name=name)

    model.compile(loss=BCE, optimizer='adam', jit_compile=True, metrics=[MMAE])

    history = model.fit(DataWrapper(train,batch_size),
        validation_data=DataWrapper(val,batch_size,False),
        epochs=10000,
        verbose=1,
        callbacks = [save,stop],
    )