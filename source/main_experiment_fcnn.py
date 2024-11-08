from model_fcnn import *
from DataGeneratorFCNN import DataGenerator
from tensorflow.keras.optimizers import Adam

gen = DataGenerator(**props)
train, val, test = gen.getData()

print(train[0].shape)
print(train[1].shape)
print(train[3].shape)

batch_size = 1

optimizer = Adam(learning_rate=0.0001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FCNN.weights.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel(train[0].shape[1:])

model.compile(loss=MSE, optimizer=optimizer, jit_compile=True, metrics=[MAE,MAX])

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=100,
    verbose=1,
    callbacks = [stop,save]
)

showResults(model, gen)