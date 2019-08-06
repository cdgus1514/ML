

from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)


# 데이터 로드
dataset = numpy.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
# dataset = numpy.loadtxt("/content/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# 모델설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# 모델 컴파일
## 분류모델 >> categorical_crossentropy
## 이진분류모델 >> binary_crossentropy
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# 모델 실행
model.fit(X, Y, epochs=200, batch_size=10)


# 결과
print("\nAccuracy >> %.4f" % (model.evaluate(X, Y)[1]))