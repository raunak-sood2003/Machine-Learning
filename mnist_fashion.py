#MNIST fashion model
#Imports
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, BatchNormalization

#Dataset
dataset = fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = dataset
X_train = X_train.reshape(60000, 28,28,1)
X_test = X_test.reshape(10000, 28,28,1)
#Model
model = Sequential()
model.add(Conv2D(64, input_shape = (28,28,1), kernel_size = 2, activation = 'relu'))
model.add(AveragePooling2D(pool_size = 2))
model.add(BatchNormalization())
model.add(Conv2D(32, input_shape = (14,14,1), kernel_size = 2, activation = 'relu'))
model.add(AveragePooling2D(pool_size = 2))
model.add(Conv2D(16, input_shape = (7,7,1), kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 3)
