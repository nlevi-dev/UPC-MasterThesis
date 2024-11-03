from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam
from DataGeneratorFFN import DataGenerator

batch_size = 100000

for i in range(7):
    props['single'] = i
    gen = DataGenerator(**props)
    train, val, test = gen.getData()

    name = 'FFN_chain_{}'.format(i)
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