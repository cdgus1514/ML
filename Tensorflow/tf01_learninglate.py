# 1. 학습데이터
import numpy as np

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])


# 2. 모델생성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim=1, activation="relu"))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))


#3. 훈련
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam

# op = Adam(lr=0.006)
# op = SGD(lr=0.01)
# op = RMSprop(lr=0.004)
# op = Adagrad(lr=0.05)
# op = Adadelta(lr=0.7)
# op = Adamax(lr=0.05)
op = Nadam(lr=0.01)
# model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.compile(loss="mse", optimizer=op, metrics=["mse"])
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가예측
mse, _ = model.evaluate(x, y, batch_size=1)
print("mse : ", mse)


pred1 = model.predict([1.5, 2.5, 3.5])
print(pred1)