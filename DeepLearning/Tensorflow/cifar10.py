#Imports
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard = TensorBoard(log_dir = run_logdir, histogram_freq=1)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean_train = np.mean(X_train, axis = (0,1,2,3))
std_train = np.std(X_train, axis = (0,1,2,3))
mean_test = np.mean(X_test, axis = (0,1,2,3))
std_test = np.std(X_test, axis = (0,1,2,3))
X_train = (X_train-mean_train)/(std_train)
X_test = (X_test - mean_test)/(std_test)

model = Sequential()

model.add(Conv2D(64, input_shape = (32,32,3), kernel_initializer = tf.keras.initializers.he_normal(seed = 3),
                 strides = (2,2), kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 2, kernel_initializer = tf.keras.initializers.he_normal(seed = 3),
                 strides = (2,2), activation = 'relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 1, batch_size = 64, validation_split = 0.3, callbacks = [tensorboard])
