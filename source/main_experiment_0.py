import os, warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from DataGenerator import DataGenerator
from visual import showSlices

gen = DataGenerator()
train, val, test = gen.getData()

model = Sequential(name='FFN')
model.add(Input((train[0].shape[1],)))
model.add(Dense(512,activation='silu'))
model.add(Dense(512,activation='silu'))
model.add(Dense(512,activation='silu'))
model.add(Dense(256,activation='silu'))
model.add(Dense(128,activation='silu'))
model.add(Dense(64,activation='silu'))
model.add(Dense(train[1].shape[1],activation='softmax'))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy, optimizer='adam', jit_compile=True)
history = model.fit(train[0], train[1],
    validation_data=val,
    batch_size=10000,
    epochs=10,
    verbose=1,
)

for t in test[0:1]:
    p = model.predict(t[0])
    showSlices(t[3],t[2](t[1]),title='original')
    showSlices(t[3],t[2](p),title='predicted')

