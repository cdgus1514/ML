import numpy as np

x_train = np.arange(1, 101)
y_train = np.arange(501, 601)
x_test = np.arange(1001, 1101)
y_test = np.arange(1101, 1201)

q = np.arange(1138, 1169)


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(10, input_dim=1, activation="relu"))
model.add(Dense(23))
model.add(Dense(50))
model.add(Dense(120))
model.add(Dense(230))
model.add(Dense(690))
model.add(Dense(370))
model.add(Dense(180))
model.add(Dense(75))
model.add(Dense(35))
model.add(Dense(17))
model.add(Dense(3))
model.add(Dense(1))


model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=200)
model.fit(x_train, y_train, epochs=200, batch_size=3)


# loss, acc = model.evaluate(x_test, y_test)
loss, acc = model.evaluate(x_test, y_test, batch_size=3)


y_predic = model.predict(q)
print(y_predic)

print("acc : ", acc)