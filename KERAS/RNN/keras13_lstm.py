from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

print("x.shape : ", x.shape)    # (4,3)
print("y.shape : ", y.shape)    # (4,)


## debug ###
print("x shape[0] >> ", x.shape[0]) # 4
print("x shape[1] >> ", x.shape[1]) # 3


# LSTM >> 몇개씩 잘라서 작업 할 것인지
x = x.reshape((x.shape[0], x.shape[1], 1))  # (4,3) >> (4,3,1)
print("reshape x.shape : ", x.shape)




# 2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3,1), activation="relu"))   # 컬럼개수, 사용할 개수 >> ((n,3), 1)
model.add(Dense(100))
model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
# model.add(Dense(100))
model.add(Dense(1))

# model.summary()


# 3. 실행
model.compile(optimizer="adam", loss="mse")
model.fit(x, y, epochs=100)


x_input = array([60,70,80])
x_input = x_input.reshape(1,3,1)

yhat = model.predict(x_input)
print(yhat)

