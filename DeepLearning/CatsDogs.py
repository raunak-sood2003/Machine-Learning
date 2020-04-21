import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Dropout
import random

data = 'C:/Users/rrsoo/AdvancedStuff/Cats and Dogs/train'
catOrDog = ['Dog', 'Cat']
train = []
for category in catOrDog:
    directory = os.path.join(data, category)
    categoryIndex = catOrDog.index(category)
    for img in os.listdir(directory):
        try:
            arr = cv2.imread(os.path.join(directory,img),cv2.IMREAD_COLOR)
            new_arr = cv2.resize(arr,(224,224))
            train.append([new_arr, categoryIndex])
        except Exception as e:
            pass

random.shuffle(train)
X = []
y = []
for lis in train:
    X.append(lis[0])
    y.append(lis[1])

X = np.array(X).reshape(-1,224,224,3)
y = np.array(y).reshape(25000,)

model = Sequential()

model.add(Conv2D(64, input_shape = (224,224,3), kernel_initializer = 'he_normal',
                 strides = (2,2), kernel_size = 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))


model.add(Conv2D(32, kernel_size = 2, kernel_initializer = 'he_normal',
                 strides = (2,2), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32, activation = 'relu', kernel_initializer= 'he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X, y, epochs = 5, batch_size = 32, validation_split = 0.2)

#93.7% accuracy



