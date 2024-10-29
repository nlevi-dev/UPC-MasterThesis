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

train_y = np.array(train[1],np.float32)
std_mask = np.sum(train_y, axis=-1)
summed = np.sum(train_y,axis=0)
summed = np.sum(summed,axis=0)
summed = np.sum(summed,axis=0)
summed = np.sum(summed,axis=0)
mean = summed/np.sum(train_y)
mean = np.repeat(np.expand_dims(mean,0),train_y.shape[-2],0)
mean = np.repeat(np.expand_dims(mean,0),train_y.shape[-3],0)
mean = np.repeat(np.expand_dims(mean,0),train_y.shape[-4],0)
mean = np.repeat(np.expand_dims(mean,0),train_y.shape[-5],0)
dev_error = np.square((train_y-mean)*np.repeat(np.expand_dims(std_mask,-1),train_y.shape[-1],-1))
dev_error = np.sum(dev_error,axis=0)
dev_error = np.sum(dev_error,axis=0)
dev_error = np.sum(dev_error,axis=0)
dev_error = np.sum(dev_error,axis=0)
std_true = np.sqrt(dev_error/np.sum(train_y))
std_mask = np.sum(std_mask, axis=0)
std_mask = np.where(std_mask >= 1, 1, 0)
del train_y; del summed; del mean; del dev_error
std_true = np.array(std_true,np.float32)
std_mask = np.array(std_mask,np.float32)

loss = CustomLoss(std_true, std_mask, diversity_weight=1)

optimizer = Adam(learning_rate=0.001)
stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
save = tf.keras.callbacks.ModelCheckpoint(filepath='data/models/FCNN.weights.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=True)

model = buildModel()

metrics = [
    MAE(weights),
    CCE(None),
    STD(std_true, std_mask),
    MAX,
]

model.compile(loss=loss, optimizer='adam', jit_compile=True, metrics=metrics)

history = model.fit(DataWrapper(train,batch_size),
    validation_data=DataWrapper(val,batch_size,False),
    epochs=100,
    verbose=1,
    callbacks = [stop,save]
)

showResults(model)