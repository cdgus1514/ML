from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt

# minmax 정규화, standard표준화
# tensorboard


# 3채널 구성된 32x32 이미지 6만장
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_CLOS = 32

# 하이퍼 파라미터 튜닝값
# best params >>  {'optimizer': 'adam', 'epochs': 50, 'drop': 0.2, 'batch_size': 15}

# 상수 정의
BATCH_SIZE = 15
NB_EPOCH = 50
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
drop = 0.2


# 데이터셋 불러오기
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print("X_train shape : ", X_train.shape)    # (50000, 32, 32, 3)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# 범주형으로 변환
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# 실수형으로 변환 및 정규화
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
# X_train /= 255
# X_test /= 255

X_train_reshape = X_train.reshape(50000, 3072)
X_test_reshape = X_test.reshape(10000, 3072)
sc = MinMaxScaler()
# sclaer.fit(X_train_reshape)
X_train_reshape = sc.fit_transform(X_train_reshape)
X_test_reshape = sc.transform(X_test_reshape)
X_train_reshape = X_train_reshape.reshape(50000, 32, 32, 3)
X_test_reshape = X_test_reshape.reshape(10000, 32, 32, 3)


# 신경망 정의
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",
                 input_shape=(IMG_ROWS, IMG_CLOS, IMG_CHANNELS)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop))

# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv2D(512, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Conv2D(512, (3, 3), padding="same"))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(drop))
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))

# model.summary()


# 학습
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# model.compile(loss="categorical_crossentropy", optimizer=OPTIM, metrics=["accuracy"])

early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=20)    # 변화값이 patience이상 변경 없을경우 중지

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    validation_split=VALIDATION_SPLIT, callbacks=[early_stopping_callback])

print("Testing...")


score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score : ", score[0])
print("Test accuracy : ", score[1])

'''
Test score :  1.3268471739068628
Test accuracy :  0.6678000173196197
'''
