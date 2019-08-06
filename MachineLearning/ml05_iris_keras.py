import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy as np



## 데이터 로드
# dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
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
print("x_test shape >> ", x_test.shape)   # (30,4)
print("y_test shape >> ", y_test.shape)   # (30,3)



## 모델구성
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(4,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)



## 모델실행
# model.fit(x_train, y_train, epochs=100, batch_size=10)
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=10)




## 모델평가
acc, loss = model.evaluate(x_test, y_test)
print("acc >> ", acc)
print("loss >> ", loss)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.grid(True)
plt.legend(["loss", "val_loss", "acc", "val_acc"])

plt.show