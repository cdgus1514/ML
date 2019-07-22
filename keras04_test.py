import numpy as np

# 1. 학습데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()


# 2. 모델구성(레이어, 노드 개수 설정)
model.add(Dense(7, input_dim=1, activation="relu")) # import >> input_dim=1(1개의 input), relu(완전 열결 층)
model.add(Dense(13))
model.add(Dense(27))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))

model.summary() # 설정한노드 + 바이어스(편향) >> 1+1 * 5 = 10, 5+1 * 3 = 18, 3+1 * 4 = 16 ....


# 3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# 훈련실행(구성한 모델에 x,y 데이터를 n개씩 짤라서 n번 반복 훈련)
# model.fit(x, y, epochs=20, batch_size=3)   # epochs >> 만들어준 모델링을 n회 반복
                                           # batch_size >> n개씩 짤라서 연산
model.fit(x_train, y_train, epochs=100, batch_size=3)


# 4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=3)

print("acc : ", acc)

# y값 예측 (x값 >> 훈련시킨 값, x2값 >> 훈련시킨 모델에 새로운 데이터) >> acc(분류모델용, 근사값을 이용해 분류), predict()
#y_predict = model.predict(x)
y_predict = model.predict(x_test)

print(y_predict)