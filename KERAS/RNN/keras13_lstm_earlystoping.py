from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import random

random.seed(1377)

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print("x.shape : ", x.shape)
print("y.shape : ", y.shape)


# LSTM >> 몇개씩 잘라서 작업 할 것인지
x = x.reshape((x.shape[0], x.shape[1], 1))
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

## earlystopping >> epochs 제한 필요없음
## monitor 옵션 >> patience 개수만큼 변화 없을 시 학습정지
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor="loss", patience=30, mode="auto")
hist = model.fit(x, y, epochs=10000, verbose=1, callbacks=[early_stopping])


x_input = array([25,35,45])
x_input = x_input.reshape(1,3,1)

yhat = model.predict(x_input, verbose=2)
print(yhat)
