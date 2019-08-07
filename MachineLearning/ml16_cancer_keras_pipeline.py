from sklearn.datasets import load_breast_cancer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings
warnings.filterwarnings("ignore")

cancer = load_breast_cancer()   #분류

x = cancer.data
y = cancer.target
print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

## 데이터 분할
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)
print("x_train shape >> ", x_train.shape)   # (455,30)
print("y_train shape >> ", y_train.shape)   # (455,)
print("x_test shape >> ", x_test.shape)     # (114,30)
print("y_test shape >> ", y_test.shape)     # (114,)



## 모델
def bulid_model(keep_prob=0.5, optimizer="adam"):
    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(30,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


## 하이퍼파라미터
def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size":batches, "optimizer":optimizer, "keep_prob":dropout}



## 튜닝
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=bulid_model, verbose=1)

hyperparameters = create_hyperparameters()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
pipe.fit(x_train, y_train)
print("테스트 점수 >> ", pipe.score(x_test, y_test))



from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

kfold_cv = KFold(n_splits=5, shuffle=True)
search = RandomizedSearchCV(pipe.get_params().keys(), hyperparameters, cv=kfold_cv)

search.fit(x_train, y_train)

print(search.best_params_)