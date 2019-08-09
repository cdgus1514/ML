from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf


# 데이터 불러온 후 전처리
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# import matplotlib.pyplot as plt
# digit = X_train[88]
# plt.show(digit, cmap=plt.cm.binary)
# plt.show()
X_train = X_train[:300]
Y_train = Y_train[:300]
# X_test = X_test[:10000]
# Y_test = Y_test[:10000]



X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype("float32")/255  # (60000, 28,28) >> (60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype("float32")/255     # 1픽셀에 255 /255 >> 전처리 실행
print(Y_train.shape)    # (60000, )
print(Y_test.shape)     # (10000, )

## onehot encoding
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(X_train.shape)  # (60000,28,28,1)
print(X_test.shape)   # (10000,28,28,1)
print(Y_train.shape)    # (60000, 10)
print(Y_test.shape)     # (10000, 10)



# 컨볼루션 신경망 설정
# 데이터가 적을 경우 >> 개수 증가 or 노드수 줄임
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), padding="same", activation="relu"))
model.add(Conv2D(64,(3,3), padding="same", activation="relu"))
model.add(Conv2D(128,(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3), padding="same", activation="relu"))
model.add(Conv2D(512,(3,3), padding="same", activation="relu"))
model.add(Conv2D(1024,(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())    # 1차원으로 변환

## 컨볼루션 신경망 히든레이어
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))  # 분류모델(지정된 값으로만 출력) 마지막 활성함수는 softmax를 사용해햐함
                                            # 출력 10 >> 0 ~ 9 중 1개 출력

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])



# 모델 최적화
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)    # 변화값이 patience이상 변경 없을경우 중지


# 60000개 새로운 이미지 생성, 훈련 실행
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.02, height_shift_range=0.02, horizontal_flip=True)

model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train)//32, epochs=200, validation_data=(X_test, Y_test), verbose=1)


# 테스트 정확성
## 분류모델은 accuracy 사용
print("\nTest Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))