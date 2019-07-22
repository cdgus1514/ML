import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(180, input_dim=1, activation="relu"))
model.add(Dense(1))

#3. 훈련
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)