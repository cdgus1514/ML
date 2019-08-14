from keras.applications import VGG19
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.models import Sequential

import numpy as np


# 1. 데이터
mnist_load = np.load("mnist_train.npz")
x_train = mnist_load["x_train"]
y_train = mnist_load["y_train"]
mnist_load = np.load("mnist_test.npz")
x_test = mnist_load["x_test"]
y_test = mnist_load["y_test"]

print("x_train shape >> ", x_train.shape)
print("x_test shape >> ", x_test.shape)
print("y_train shape >> ", y_train.shape)
print("y_test shape >> ", y_test.shape)

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)


## one hot encoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


### mnist 1채널 >> 3채널
x_train = np.dstack([x_train]*3)
x_test = np.dstack([x_test]*3)
x_train = x_train.reshape(60000,28,28,3)
x_test = x_test.reshape(10000,28,28,3)

print("x_train shape >> ", x_train.shape)
print("x_test shape >> ", x_test.shape)


### vgg16 input_size에 맞게 크기 조정
from keras.preprocessing.image import img_to_array, array_to_img
x_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_train])
x_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test])
print("x_train shape >> ", x_train.shape)
print("x_test shape >> ", x_test.shape)


### 전처리
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")


print("######## 최종 shape 확인 ########")
print("x_train shape >> ", x_train.shape)
print("x_test shape >> ", x_test.shape)
print("y_train shape >> ", y_train.shape)
print("y_test shape >> ", y_test.shape)



# 2. 무델
conv_base = VGG19(weights="imagenet", include_top=False, input_shape=(48,48,3))

model = Sequential()

model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)


print("\nTest Accuracy : %.4f" % (model.evaluate(x_test, y_test)[1]))

# Test Accuracy : 0.9986