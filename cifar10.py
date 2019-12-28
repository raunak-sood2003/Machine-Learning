from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#Loading data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
mean_train = np.mean(X_train, axis = (0,1,2,3))
std_train = np.std(X_train, axis = (0,1,2,3))
mean_test = np.mean(X_test, axis = (0,1,2,3))
std_test = np.std(X_test, axis = (0,1,2,3))
X_train = (X_train-mean_train)/(std_train)
X_test = (X_test - mean_test)/(std_test)

model = Sequential()
model.add(Conv2D(64, input_shape = (32,32,3), kernel_size = 3, activation = 'relu'))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 5, batch_size = 64)