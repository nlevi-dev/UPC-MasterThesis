import os, warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from DataGenerator import DataGenerator

gen = DataGenerator()
train, test, val = gen.getData()

model = Sequential(name='FFN')
model.add(Input((train[0].shape[1],)))
model.add(Dense(512,activation='silu'))
model.add(Dense(512,activation='silu'))
model.add(Dense(512,activation='silu'))
model.add(Dense(256,activation='silu'))
model.add(Dense(128,activation='silu'))
model.add(Dense(64,activation='silu'))
model.add(Dense(train[1].shape[1],activation='softmax'))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy, optimizer='adam')
history = model.fit(train[0], train[1],
    validation_data=val,
    batch_size=10000,
    epochs=20,
    verbose=1,
)