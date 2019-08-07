from sklearn.datasets import load_breast_cancer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators

cancer = load_breast_cancer()   #분류

x = cancer.data
y = cancer.target
print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

## 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)
print("x_train shape >> ", x_train.shape)   # (455,30)
print("y_train shape >> ", y_train.shape)   # (455,)
print("x_test shape >> ", x_test.shape)     # (114,30)
print("y_test shape >> ", y_test.shape)     # (114,)



## 모델
def bulid_model(keep_prob=0.2, optimizer="adam"):
    model = Sequential()

    model.add(Dense(128, input_dim=30, activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(32))
    model.add(Dropout(keep_prob))
    model.add(Dense(32))
    model.add(Dropout(keep_prob))
    # model.add(Dense(2, activation="softmax"))
    model.add(Dense(1, activation="sigmoid"))

    # model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model




## 튜닝
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


kfold_cv = KFold(n_splits=5, shuffle=True)
model = KerasClassifier(build_fn=bulid_model)

prameter = {
    "model__batch_size": [5,15,25,35,55],
    "model__optimizer": ["adam", "adadelta", "rmsprop"],
    "model__keep_prob": [0, 0.2, 0.25, 0.5],
    "model__epochs": [100, 200, 500]
}

pipe = Pipeline([("scaler", MinMaxScaler()), ("model", model)])

search = RandomizedSearchCV(pipe, prameter, cv=kfold_cv)
search.fit(x_train, y_train)

print(search.best_params_)

from sklearn.metrics import accuracy_score
y_pred = search.predict(x_test)
print('정답률 >> ', accuracy_score(y_test, y_pred))