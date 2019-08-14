from keras.applications import VGG16
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model

import numpy as np
import matplotlib.pyplot as plt


# 1. 데이터
cifar10_load = np.load("cifar10_train.npz")
x_train = cifar10_load["x_train"]
y_train = cifar10_load["y_train"]
cifar10_load = np.load("cifar10_test.npz")
x_test = cifar10_load["x_test"]
y_test = cifar10_load["y_test"]


## 실수형으로 변환 및 정규화
from sklearn.preprocessing import MinMaxScaler, StandardScaler
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)


print("######## 최종 shape 확인 ########")
print("x_train shape >> ", x_train.shape)
print("x_test shape >> ", x_test.shape)
print("y_train shape >> ", y_train.shape)
print("y_test shape >> ", y_test.shape)



# 2. 무델
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))

model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)