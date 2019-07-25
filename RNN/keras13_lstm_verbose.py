from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import random

random.seed(1377)

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)    # 결과값의 개수


# 몇개씩 작업을 할것인지 설정
x = x.reshape((x.shape[0], x.shape[1], 1))  # (4,3,1) >> (3,1)

print("reshape x.shape : ", x.shape)


# 2. 모델구성
model = Sequential()
model.add(LSTM(10, activation="relu", input_shape=(3,1)))   # 컬럼개수, 사용할 개수 >> ((n,3), 1)
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()


# 3. 실행
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=1000, verbose=0) # >> 0 출력 X, 1 자세히, 2 loss, 3/4 개수만


x_input = array([25,35,45])
x_input = x_input.reshape(1,3,1)

yhat = model.predict(x_input, verbose=2)
print(yhat)

