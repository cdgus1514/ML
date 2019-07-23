import numpy as np

# 1. 학습데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# 2. 모델생성(레이어, 노드 개수 설정)
model.add(Dense(5, input_dim=1, activation="relu")) # import >> input_dim=1(1개의 input)
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


#3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, epochs=100, batch_size=1)   # 만들어준 모델링을 100회 반복

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size=1)

print("acc : ", acc)