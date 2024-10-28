from model_fcnn import *
from tensorflow.keras.optimizers import Adam

gen = DataGenerator(**props)
train, val, test = gen.getData()

weights = []
for i in range(train[1].shape[-1]):
    weights.append(np.count_nonzero(train[1][:,:,:,:,i]))
weights = np.array(weights,np.float32)
weights /= np.sum(weights)
weights = weights*-1+1
weights /= np.sum(weights)
weights *= len(weights)

batch_size = 1

loss = CustomLoss(weights,train[1].shape,batch_size,diversity_weight=1)
optimizer = Adam(learning_rate=0.001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FCNN.weights.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel()

metrics = [
    MAE(weights,train[1].shape,batch_size),
    MSE(weights,train[1].shape,batch_size),
    CCE(None,train[1].shape,batch_size),
    MAX,
    STD,
]

model.compile(loss=loss, optimizer='adam', jit_compile=True, metrics=metrics)

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=100,
    verbose=1,
    callbacks = [stop,save]
)

showResults(model)