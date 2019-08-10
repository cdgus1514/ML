from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np

# 상수 정의
BATCH_SIZE = 200
NB_EPOCH = 1000
NB_CLASSES = 10
VERBOSE = 1


# 1. 데이터

# 데이터 로드
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# 데이터 분할(300개씩)
print("X_train shape >> ", X_train.shape)   # (50000, 32, 32, 3)
print("Y_train shape >> ", Y_train.shape)   # (50000, 1)
print("X_test shape : ", X_test.shape)      # (10000, 32, 32, 3)
print("Y_test shape : ", Y_test.shape)      # (10000, 1)
print("======================================================")

X_train, X_train300, Y_train, Y_train300 = train_test_split(
    X_train, Y_train, test_size=0.006, shuffle=True)

print("X_train300 shape >> ", X_train300.shape)   # (300, 32, 32, 3)
print("Y_train300 shape >> ", Y_train300.shape)   # (300, 1)

# X_test, X_test300, Y_test, Y_test300 = train_test_split(X_test, Y_test, test_size=0.03, shuffle=True)
# print("X_test300 shape >> ", X_test300.shape)   # (300, 32, 32, 3)
# print("Y_test300 shape >> ", Y_test300.shape)   # (300, 1)

# 범주형으로 변환(one hot encoding)
Y_train300 = np_utils.to_categorical(Y_train300, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)
# Y_test300 = np_utils.to_categorical(Y_test300, NB_CLASSES)

# 실수형으로 변환
X_train300 = X_train300.astype("float32")
X_test = X_test.astype("float32")
# X_test300 = X_test300.astype("float32")

# 정규화
X_train300 = X_train300.reshape(300, 3072)
X_test = X_test.reshape(10000, 3072)
# X_test_reshape = X_test300.reshape(300, 3072)

sc = MinMaxScaler()
# sc = StandardScaler()
X_train300 = sc.fit_transform(X_train300)
X_test300 = sc.transform(X_test)
X_train300 = X_train300.reshape(300, 32, 32, 3)
X_test = X_test.reshape(10000, 32, 32, 3)

# 최종 데이터셋 차원 확인
print("=======================================")
print("X_train300 shape >> ", X_train300.shape)   # (300, 32, 32, 3)
print("Y_train300 shape >> ", Y_train300.shape)   # (300, 10)
print("X_test shape >> ", X_test.shape)           # (10000, 32, 32, 3)
print("Y_test shape >> ", Y_test.shape)           # (10000, 10)
print("=======================================")


# 2. 모델
def bulid_model(optimizer="adam", drop=0.2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop))

    # model.add(Conv2D(128, (3,3), padding="same"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, (3,3), padding="same"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(drop))

    # model.add(Conv2D(512, (3,3), padding="same"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(512, (3,3), padding="same"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(drop))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(drop))

    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    print("Generate start...")
    model.fit_generator(datagen.flow(X_train300, Y_train300,
                                     batch_size=200), steps_per_epoch=1, epochs=1)
    print("end...")

    return model


# 이미지 데이터 증폭

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True)


# 3. 튜닝
k_cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=bulid_model)

# 파라미터 설정
parameters = {
    "batch_size": [5, 15, 55],
    "optimizer": ["adam", "adadelta", "rmsprop"],
    "drop": [0, 0.2, 0.5],
    "epochs": [10, 20, 50]

    # "model__batch_size": [5,15,55],
    # "model__optimizer": ["adam", "adadelta", "rmsprop"],
    # "model__drop": [0,0.2,0.5],
    # "model__epochs": [50, 100,200]
}


# pipe = Pipeline([
#     ("scaler", MinMaxScaler()), ("model", model)
# ])

search = RandomizedSearchCV(model, parameters, cv=k_cv)
# search = RandomizedSearchCV(pipe, parameters, cv=k_cv)
search.fit(X_train300, Y_train300)
print("best estimator >> ", search.best_estimator_)
print("best params >> ", search.best_params_)
print("best score >> ", search.best_score_)


# # 4. 평가
# Y_pred = search.predict(X_test300)
# print("최종 정답률 >> ", accuracy_score(Y_test, Y_pred))
# last_score = search.score(Y_test, Y_test)
# print("최종 정답률 >> ", last_score)


# # RMSE


# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))


# print("RMSE : ", RMSE(Y_test, Y_pred))


'''
## stop
stop = EarlyStopping(monitor="loss", patience=5)

## generator
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.02, height_shift_range=0.02, horizontal_flip=True)

# model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE, callbacks=[stop])
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=BATCH_SIZE), steps_per_epoch=300, epochs=200, callbacks=[stop])

print("Testing...")


score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print("\nTest score : ", score[0])
print("Test accuracy : ", score[1])


## acc
print("\nTest Accuracy : %.4f" % (model.evaluate(X_test, Y_test)[1]))
'''
