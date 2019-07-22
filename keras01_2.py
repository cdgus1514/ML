import numpy as np

# 1. 학습데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
x2 = np.array([4,5,6])


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 2. 모델생성(레이어, 노드 개수 설정)
model.add(Dense(10, input_dim=1, activation="relu")) # import >> input_dim=1(1개의 input)
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(50))
# model.add(Dense(7))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


#3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, epochs=20, batch_size=3)   # 만들어준 모델링을 n회 반복

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size=3)

print("acc : ", acc)

# y값 예측 (x값 >> 훈련시킨 값)
#y_predict = model.predict(x)
y_predict = model.predict(x2)

print(y_predict)