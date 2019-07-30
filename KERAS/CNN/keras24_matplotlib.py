from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf


# 데이터 불러온 후 전처리
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
import matplotlib.pyplot as plt
# digit = X_train[88]
# plt.show(digit, cmap=plt.cm.binary)
# plt.show()
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
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())    # 1차원으로 변환

## 컨볼루션 싱경망 히든레이어
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))  # 분류모델(지정된 값으로만 출력) 마지막 활성함수는 softmax를 사용해햐함
                                            # 출력 10 >> 0 ~ 9 중 1개 출력

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



# 모델 최적화
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)    # 변화값 10번이상 변경 없을경우 중지



# 모델 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=200, verbose=1, callbacks=[early_stopping_callback])



# 테스트 정확성
## 분류모델은 accuracy 사용
print("\nTest Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))

print(history.history.keys()) # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])


import matplotlib.pyplot as plt
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model loss, accuracy")
plt.ylabel("loss, acc")
plt.xlabel("epochs")
plt.legend(["train loss", "test loss", "train acc", "test acc"], loc="upper left")
plt.show()

