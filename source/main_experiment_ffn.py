from model_ffn import *
from DataGeneratorFFN import DataGenerator
from tensorflow.keras.optimizers import Adam

gen = DataGenerator(**props)
train, val, test = gen.getData()

print(train[0].shape)
print(train[1].shape)

batch_size = 10000

optimizer = Adam(learning_rate=0.00001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FFN.weights.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel(train[0].shape[-1])

model.compile(loss=MSE, optimizer=optimizer, jit_compile=True, metrics=[MAE,STD])

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=10000,
    verbose=1,
    callbacks = [stop,save],
)

showResults(model, gen)