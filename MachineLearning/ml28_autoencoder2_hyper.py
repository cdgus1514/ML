from keras.datasets import mnist
import numpy as np

# 비지도학습 데이터 불러오기
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)    # (60000,784)
print(x_test.shape)     # (10000,784)



# 2. 모델
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model


def bulid_model(optimizer="adadelta", drop=0.2, epochs=10, batch_size=128):
    # 인코딩 데이터 크기 설정
    encoding_dim = 32

    # 입력 플레이스홀더
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation="relu")(input_img)

    hidden = Dense(128, activation="relu")(encoded)
    Dropout(drop)(hidden)
    hidden = Dense(128, activation="relu")(hidden)
    hidden = Dense(128, activation="relu")(hidden)
    Dropout(drop)(hidden)
    hidden = Dense(64, activation="relu")(hidden)
    hidden = Dense(64, activation="relu")(hidden)
    hidden = Dense(64, activation="relu")(hidden)
    Dropout(drop)(hidden)
    hidden = Dense(32, activation="relu")(hidden)
    hidden = Dense(32, activation="relu")(hidden)

    decoded = Dense(784, activation="sigmoid")(hidden)

    autoencoder = Model(input_img, decoded)     # 784 >> 32 >> 784
    # 인코딩된 입력의 표현으로 매핑
    encoder = Model(input_img, encoded)         # 784 >> 32

    encoded_input = Input(shape=(encoding_dim,))
    decoded_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoded_layer(encoded_input))

    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
#     autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))


    return autoencoder


## 튜닝
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

kfold_cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=bulid_model)

parameter = {
    "batch_size": [128,256,512],
    "optimizer": ["adam", "adadelta", "rmsprop"],
    "drop": [0,0.2,0.5],
    "epochs": [10,50]
}

# 같은 값의 x, y를 모델에 넣음 >> 다른 값 출력
search = RandomizedSearchCV(model, parameter, cv=kfold_cv)
search.fit(x_train, x_train)

print("최적의 파라미터 >> ", search.best_params_)
# 최적의 파라미터 >>  {'optimizer': 'adam', 'epochs': 50, 'drop': 0.5, 'batch_size': 256}