import numpy as np

# 1. 학습데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([4,5,6])


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 2. 모델구성(레이어, 노드 개수 설정)
model.add(Dense(10, input_dim=1, activation="relu")) # import >> input_dim=1(1개의 input), relu(완전 열결 층)
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(50))
# model.add(Dense(7))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


# 3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# 훈련실행(구성한 모델에 x,y 데이터를 n개씩 짤라서 n번 반복 훈련)
# model.fit(x, y, epochs=20, batch_size=3)   # epochs >> 만들어준 모델링을 n회 반복
                                           # batch_size >> n개씩 짤라서 연산
model.fit(x, y, epochs=20)


# 4. 평가예측
loss, acc = model.evaluate(x, y, batch_size=3)

print("acc : ", acc)

# y값 예측 (x값 >> 훈련시킨 값, x2값 >> 훈련시킨 모델에 새로운 데이터) >> acc(분류모델용, 근사값을 이용해 분류), predict()
#y_predict = model.predict(x)
y_predict = model.predict(x2)

print(y_predict)