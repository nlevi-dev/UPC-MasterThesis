from model_ffn_chain import *
from tensorflow.keras.optimizers import Adam

for i in [2,3,6]:
    props['single'] = i
    gen = DataGenerator(**props)
    train, val, test = gen.getData()

    optimizer = Adam(learning_rate=0.001)
    stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
    save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FFN_chain_{}.weights.h5'.format(i),monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

    model = buildModel('FFN_chain_{}'.format(i))

    model.compile(loss=BCE, optimizer='adam', jit_compile=True, metrics=[MAE])

    history = model.fit(DataWrapper(train,batch_size),
        validation_data=DataWrapper(val,batch_size,False),
        epochs=10000,
        verbose=1,
        callbacks = [save,stop],
    )