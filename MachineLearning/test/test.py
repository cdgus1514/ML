# 데이터
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer() # 분류.
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 66)
print('x',x_train.shape, x_test.shape)
print('y',y_train.shape, y_test.shape)


# 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[-1], 1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[-1], 1)

input_size = x_train.shape[1] # 30
output_size = 1


def build_network(optimizer = 'adam', drop = 0.2):
    model = Sequential()
    model.add(Dense(128, input_dim= 30, activation = 'relu'))
    # model.add(LSTM(64, input_shape= input_size, activation = 'relu'))
    model.add(Dropout(drop))
    model.add(Dense(32))
    model.add(Dropout(drop))
    model.add(Dense(32))
    model.add(Dropout(drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    return model

# 교차 검증
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler

kfold = KFold(n_splits = 5, shuffle= True)
model = KerasClassifier(build_fn = build_network)
parmeters = {
    'model__optimizer': ['adam', 'adadelta', 'rmsprop'],
    'model__batch_size': [1,2,3],
    'model__epochs': [100,200,300],
    'model__drop': [0, 0.25, 0.5]
}
pipe = Pipeline([('scaler', MinMaxScaler()), ('model', model)])
search = RandomizedSearchCV(pipe, parmeters, cv = kfold)
search.fit(x_train, y_train)

# 결과
print('>> 최적의 매개 변수\n', search.best_params_)
y_pred = search.predict(x_test)

from sklearn.metrics import accuracy_score
print('>> 최종 정답률:', accuracy_score(y_test, y_pred))


# >> 최적의 매개 변수
#  {'model__optimizer': 'rmsprop', 'model__epochs': 100, 'model__drop': 0, 'model__batch_size': 1}
#>> 최종 정답률: 0.9210526315789473
