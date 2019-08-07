# from keras.models import Sequential, Model
# from keras.layers import Dense, LSTM, Dropout, Input
# import numpy as np

# from keras.datasets import mnist
# from keras.utils import np_utils
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, LSTM
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# import numpy
# import os
# import tensorflow as tf


# # 데이터 불러온 후 전처리
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# import matplotlib.pyplot as plt
# # digit = X_train[88]
# # plt.show(digit, cmap=plt.cm.binary)
# # plt.show()

# X_train = X_train.reshape(X_train.shape[0], 28,28,1).astype("float32")/255  # (60000, 28,28) >> (60000, 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28,28,1).astype("float32")/255     # 1픽셀에 255 /255 >> 전처리 실행
# print(Y_train.shape)    # (60000, )
# print(Y_test.shape)     # (10000, )

# ## onehot encoding
# Y_train = np_utils.to_categorical(Y_train)
# Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape)  # (60000,28,28,1)
# print(X_test.shape)   # (10000,28,28,1)
# print(Y_train.shape)    # (60000, 10)
# print(Y_test.shape)     # (10000, 10)

# def bulid_network(keep_prob=0.5, optimizer="adam"):
#     inputs = Input(shape=(784, ), name="input")
#     x = Dense(512, activation="relu", name="hidden1")(inputs)
#     x = Dropout(keep_prob)(x)
#     x = Dense(256, activation="relu", name="hidden2")(inputs)
#     x = Dropout(keep_prob)(x)
#     x = Dense(128, activation="relu", name="hidden3")(inputs)
#     x = Dropout(keep_prob)(x)

#     prediction = Dense(10, activation="softmax", name="output")(x)
#     model = Model(input=input, output=prediction)
#     model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

#     return model
    


# def create_hyperparameters():
#     batches = [10,20,30,40,50]
#     optimizers = ["rmsprop", "adam", "adadelta"]
#     dropout = np.linspace(0.1, 0.5, 5)
#     return{"batch_size":batches, "optimizer":optimizers, "keep_prob":dropout}




# ## 교차검증을 위한 sklearn 사용
# from keras.wrappers.scikit_learn import KerasClassifier

# model = KerasClassifier(build_fn=bulid_network, verbose=1)

# hyperparameters = create_hyperparameters()


# from sklearn.model_selection import RandomizedSearchCV
# # 작업 10회 실행, 3겹 교차검증 실행
# search = RandomizedSearchCV(estimator=model, param_distriburions=hyperparameters, n_iter=10, n_jobs=1, cv=3, verbose=1)
# search.fit(X_train, Y_train)

# print(search.best_params_)

#mnist : 분류 모델

from keras.datasets import mnist #6만장에 대한 데이터
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy
import os
import tensorflow as tf


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#데이터 불러오기
#스칼라: 데이터 한 개 #벡터: 연결된 데이터
#0과 1사이로 MinMaxscaler 한 것 /255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
# X_train = X_train.reshape(X_train.shape[0], 28*28).astype('float32') / 255
# X_test = X_test.reshape(X_test.shape[0], 28*28).astype('float32') / 255
print(Y_train.shape)
print(Y_test.shape)
Y_train = np_utils.to_categorical(Y_train) #분류됨
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape) #(60000, 10) 뒤에 10은 데이터가 10. 3이면 0001 4면 00001 
print(Y_test.shape)
#OneHot Incoding -- 카테고리컬이 만든
# print(X_train.shape)
# print(X_test.shape)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):
    model = Sequential()
    
    # inputs = Input(shape=(28*28, ), name='input')
    # x = Dense(512, activation='relu', name='hidden1')(inputs)
    # x = Dropout(keep_prob)(x)
    # x = Dense(256, activation='relu', name='hidden2')(inputs)
    # x = Dropout(keep_prob)(x)
    # x = Dense(128, activation='relu', name='hidden3')(inputs) #얘만 돌아감
    # x = Dropout(keep_prob)(x)
    # prediction = Dense(10, activation='softmax', name='output')(x)
    # model = Model(inputs=inputs, outputs=prediction)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation="relu"))
    model.add(Conv2D(64,(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(keep_prob))
    model.add(Flatten())    # 1차원으로 변환

    ## 컨볼루션 싱경망 히든레이어
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizer, "keep_prob":dropout}

    
from keras.wrappers.scikit_learn import KerasClassifier #사이킷런과 호환하도록 한다 #분류(mnist같은)
model = KerasClassifier(build_fn=build_network, verbose=1)

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV
search = GridSearchCV(model, hyperparameters, cv=3)

# search.fit(data["X_train"], data["y_train"])
search.fit(X_train, Y_train)

print(search.best_params_)


'''
{'optimizer': 'adam', 'keep_prob': 0.1, 'batch_size': 20}
'''