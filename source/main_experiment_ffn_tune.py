from model_ffn import *
from tensorflow.keras.optimizers import Adam

gen = DataGenerator(**props)
train, val, test = gen.getData()

optimizer = Adam(learning_rate=0.0001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FFN_tuned.weights.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel()

model.compile(loss=MSE, optimizer='adam', jit_compile=True, metrics=[CCE,MAE,MSE])

model.load_weights('data/models/FFN.weights.h5')

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=10000,
    verbose=1,
    callbacks = [save,stop],
)

showResults(model, threshold=props['threshold'])