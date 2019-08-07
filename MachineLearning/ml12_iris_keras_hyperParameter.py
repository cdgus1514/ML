import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical




## 데이터 로드
iris_data = pd.read_csv("/content/iris.csv", names=["SepalLenght", "SepalWidth", "PetalLenght", "petalWidth", "Name"], encoding="utf-8")
# print("iris shape >> ", iris_data.shape)    #(150,5)
print(iris_data["Name"].unique())



## 문자열로된 이름에 번호(LabelEncoder)를 붙이고 그 번호를 원핫인코딩 방식으로 변경
from sklearn.preprocessing import LabelEncoder
x = iris_data.iloc[:, 0:4].values
y = iris_data.iloc[:, 4].values

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
y = pd.get_dummies(y1).values

print("x shape >> ", x.shape)   # (150,4)
print("y shape >> ", y.shape)   # (150,3)



## 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size=0.8, shuffle=True)
print("x_train shape >> ", x_train.shape)   # (120,4)
print("y_train shape >> ", y_train.shape)   # (120,3)
print("x_test shape >> ", x_test.shape)     # (30,4)
print("y_test shape >> ", y_test.shape)     # (30,3)


## 모델
def bulid_model(keep_prob=0.5, optimizer="adam"):
    model = Sequential()

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(4,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(3, activation="softmax"))

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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

kfold_cv = KFold(n_splits=5, shuffle=True)
search = RandomizedSearchCV(model, hyperparameters, cv=kfold_cv)

search.fit(x_train, y_train)

print(search.best_params_)

'''
# result

'''